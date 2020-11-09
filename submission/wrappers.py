import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
import cv2


class DTPytorchWrapper:
    def __init__(self, shape=(120, 160, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])
        self.compose_obs = Compose(
            [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

    def preprocess(self, obs):
        obs = Image.fromarray(cv2.resize(obs, dsize=self.shape[0:2][::-1]))
        return obs

    def toTensor(self, obs):
        return self.compose_obs(obs)
