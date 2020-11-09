import os
import torch
from torch import nn

import numpy as np
from torchvision.models.resnet import conv3x3


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu(conv1)

        conv2 = self.conv2(x)
        conv2 = self.bn2(conv2)

        return conv1 + conv2


class Dronet(nn.Module):
    """
    A class used to define action regressor model based on Dronet arch.
    Loquercio, Antonio, et al. "Dronet: Learning to fly by driving." IEEE Robotics and Automation Letters 3.2 (2018): 1088-1095.
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images
    loss(*args)
        takes images and target action to compute the loss function used in optimization
    predict(*args)
        takes images as input and predict the action space unnormalized
    """

    def __init__(self, num_outputs=2, max_velocity=0.3, max_steering=np.pi / 2):
        """
        Parameters
        ----------
        num_outputs : int
            number of outputs of the action space (default 2)
        max_velocity : float
            the maximum velocity used by the teacher (default 0.7)
        max_steering : float
            maximum steering angle as we are predicting normalized [0-1] (default pi/2)
        """
        super(Dronet, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_outputs = num_outputs
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=[2, 2]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64, 128, stride=2),
            Flatten(),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.num_feats_extracted = 2560
        # predicting steering angle
        self.steering_angle_channel = nn.Sequential(
            nn.Linear(self.num_feats_extracted, 1)
        )

        # predicting if the bot should speed up or slow down
        self.speed_up_channel = nn.Sequential(nn.Linear(self.num_feats_extracted, 1))


        # Max steering angle, minimum velocity and maximum velocity parameters
        self.max_steering = max_steering
        self.max_velocity = max_velocity
        self.max_velocity_tensor = torch.tensor(self.max_velocity).to(self._device)
        self.min_velocity = 0.1
        self.min_velocity_tensor = torch.tensor(self.min_velocity).to(self._device)

    def forward(self, images):
        """
        Parameters
        ----------
        images : tensor
            batch of input images to get normalized predicted action
        Returns
        -------
        action: tensor
            normalized predicted action from the model
        """
        features = self.feature_extractor(images)
        steering_angle = self.steering_angle_channel(features)
        is_speed_up = self.speed_up_channel(features)
        return is_speed_up, steering_angle

    def predict(self, *args):
        """
        Parameters
        ----------
        *args : tensor
            batch of input images to get unnormalized predicted action
        Returns
        -------
        action: tensor
            action having velocity and omega of shape (batch_size, 2)
        """
        images = args[0]
        is_speed_up, steering_angle = self.forward(images)
        is_speed_up = torch.sigmoid(is_speed_up)
        v_tensor = (is_speed_up) * self.max_velocity_tensor + (
            1 - is_speed_up
        ) * self.min_velocity_tensor
        steering_angle = steering_angle * self.max_steering
        output = torch.cat((v_tensor, steering_angle), 1).squeeze().detach()
        return output.cpu().numpy().flatten()

    def close(self):
        pass

    def load(self, filename, directory, for_inference=False):
        model_path = os.path.join(directory, filename)
        self.load_state_dict(torch.load(model_path, map_location=self._device))
        if for_inference:
            self.eval()
        else:
            self.train()
