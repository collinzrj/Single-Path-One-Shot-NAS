import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(Synthesizer):
    name = 'Pattern'
    pattern_tensor: torch.Tensor = torch.tensor([
        [1., 0., 1.],
        [-10., 1., -10.],
        [-10., -10., 0.],
        [-10., 1., -10.],
        [1., 0., 1.]
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into."
    y_top = 23
    "Y coordinate to put the backdoor into."

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    resize_scale = (5, 6)
    "If the pattern is dynamically placed, resize the pattern."

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def make_pattern(self):
        full_image = torch.zeros_like(self.input_stats.average_input_values)
        full_image.fill_(self.mask_value)

        x_bot = self.x_top + self.pattern_tensor.shape[0]
        y_bot = self.y_top + self.pattern_tensor.shape[1]

        full_image[:, self.x_top:x_bot, self.y_top:y_bot] = self.pattern_tensor

        self.mask = 1 * (full_image != self.mask_value)
        min_val_pattern = self.input_stats.min_val * (1 * (full_image == 0))
        max_val_pattern = self.input_stats.max_val * (1 * (full_image == 1))
        self.pattern = min_val_pattern + max_val_pattern

        return
