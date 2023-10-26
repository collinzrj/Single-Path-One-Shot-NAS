import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.pattern_synthesizer import PatternSynthesizer

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class DynamicSynthesizer(PatternSynthesizer):
    name = 'Dynamic'

    # 5 by 5 pattern
    # pattern_tensor: torch.Tensor = torch.tensor([
    #     [1.]
    # ])
    # "Just some random 2D pattern."
    pattern_tensor_updated = None
    resize_scale = (4, 8)
    "If the pattern is dynamically placed, resize the pattern."

    def update_pattern(self):
        resize = random.randint(self.resize_scale[0], self.resize_scale[1])
        if random.random() > 0.5:
            self.pattern_tensor_updated = functional.hflip(self.pattern_tensor)
        else:
            self.pattern_tensor_updated = self.pattern_tensor
        image = transform_to_image(self.pattern_tensor_updated)
        self.pattern_tensor_updated = transform_to_tensor(functional.resize(image, [resize, resize])).squeeze()
        self.x_top = random.randint(0, self.input_stats.input_shape[1] -
                                    self.pattern_tensor_updated.shape[0] - 1)
        self.y_top = random.randint(0, self.input_stats.input_shape[2] -
                                    self.pattern_tensor_updated.shape[1] - 1)
        self.make_pattern()

    def apply_mask(self, input_tensor):
        self.update_pattern()
        return (1 - self.mask) * input_tensor + self.mask * self.pattern

    def make_pattern(self):
        full_image = torch.zeros_like(self.input_stats.average_input_values)
        full_image.fill_(self.mask_value)

        if self.pattern_tensor_updated is None:
            self.update_pattern()

        x_bot = self.x_top + self.pattern_tensor_updated.shape[0]
        y_bot = self.y_top + self.pattern_tensor_updated.shape[1]
        full_image[:, self.x_top:x_bot, self.y_top:y_bot] = self.pattern_tensor_updated

        self.mask = 1 * (full_image != self.mask_value)
        # min_val_pattern = torch.min(self.input_stats.min_val) * (1 * (full_image == 0))
        # max_val_pattern = torch.max(self.input_stats.max_val) * (1 * (full_image == 1))
        self.pattern = full_image * torch.max(self.input_stats.max_val)

        return