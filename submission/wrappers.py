import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
import cv2

class DTPytorchWrapper:
    def __init__(self, shape=(120, 160, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])
        self.compose_obs = Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
        ])

    def preprocess(self, obs):
        # from PIL import Image
        # return np.array(Image.fromarray(obs).resize(self.shape[0:2])).transpose(2, 0, 1)
        obs = cv2.resize(obs, self.shape[0:2])
        # NOTICE: OpenCV changes the order of the channels !!!
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        return obs#.transpose(2, 0, 1)

    def toTensor(self, obs):
        return self.compose_obs(obs)
