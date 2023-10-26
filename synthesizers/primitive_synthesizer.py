import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PrimitiveSynthesizer(Synthesizer):
    name = 'Primitive'

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image."

    def make_pattern(self):
        try:
            if self.params.random_seed is not None:
                print("result is a random seed", self.params.random_seed)
                torch.manual_seed(self.params.random_seed)
        except:
            print("No random_seed")
        # input_shape = self.input_stats.input_shape
        # if len(input_shape) != 3:
        #     raise ValueError("Input shape must be 3D.")
        # total_elements = input_shape[1] * input_shape[2]
        total_elements = self.input_stats.average_input_values.view(-1).shape[0]
        rand_tensor = torch.rand_like(self.input_stats.average_input_values)
        input_placeholder = (rand_tensor > 0.5) * self.input_stats.max_val + (rand_tensor <= 0.5) * self.input_stats.min_val
        cover_size = max(1, int(total_elements * self.params.backdoor_cover_percentage))
        start_index = np.random.randint(0, total_elements - cover_size - 1, size=1)[0]
        self.mask = torch.zeros_like(input_placeholder)
        # self.mask.view(self.mask.shape[0], -1)[:, start_index:start_index + cover_size] = 1
        self.mask.view(-1)[start_index:start_index + cover_size] = 1
        self.pattern = input_placeholder
