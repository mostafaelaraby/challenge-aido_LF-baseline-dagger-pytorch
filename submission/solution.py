#!/usr/bin/env python3
import os

import numpy as np
import torch

from aido_schemas import EpisodeStart, protocol_agent_DB20, PWMCommands, DB20Commands, LEDSCommands, RGB, \
    wrap_direct, Context, DB20Observations, JPGImage, logger

from model import Dronet
from wrappers import DTPytorchWrapper
from helpers import SteeringToWheelVelWrapper
from PIL import Image
import io


class PytorchRLTemplateAgent:
    def __init__(self, load_model=False, model_path=None):
        logger.info('PytorchRLTemplateAgent init')
        self.preprocessor = DTPytorchWrapper()
        self.image_size = (120,160, 3)
        self.wrapper  = SteeringToWheelVelWrapper()
        self.model = Dronet()
        self._device = self.model._device
        self.model.to(self._device)
        self.current_image = np.zeros((3,self.image_size[0],self.image_size[1]))

        if load_model:
            logger.info('PytorchRLTemplateAgent loading models')
            fp = model_path if model_path else "model.pt"
            self.model.load(fp, "models", for_inference=True)
            logger.info('PytorchRLTemplateAgent model loaded')
        logger.info('PytorchRLTemplateAgent init complete')

    def init(self, context: Context):
        available = torch.cuda.is_available()
        req = os.environ.get('AIDO_REQUIRE_GPU', None)
        context.info(f'torch.cuda.is_available = {available!r} AIDO_REQUIRE_GPU = {req!r}')
        context.info('init()')
        if available:
            i = torch.cuda.current_device()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(i)
            context.info(f'device {i} of {count}; name = {name!r}')

        else:
            if req is not None:
                msg = 'I need a GPU; bailing.'
                context.error(msg)
                raise Exception(msg)


    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: DB20Observations):
        camera: JPGImage = data.camera
        obs = jpg2rgb(camera.jpg_data)
        self.current_image = self.preprocessor.preprocess(obs)

    def compute_action(self, observation):
        observation = self.preprocessor.toTensor(observation).to(self._device).unsqueeze(0)
        action = self.model.predict(observation)
        return action.astype(float)

    def on_received_get_commands(self, context: Context):
        velocity, omega = self.compute_action(self.current_image)
        # multiplying steering angle by a gain
        omega *= 7
        pwm_left, pwm_right = self.wrapper.convert(velocity, omega)
        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = DB20Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data

def main():
    node = PytorchRLTemplateAgent(load_model=True, model_path="model_lf.pt")
    protocol = protocol_agent_DB20
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
