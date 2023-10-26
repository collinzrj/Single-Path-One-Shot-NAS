import numpy as np
import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class MemorySynthesizer(Synthesizer):
    name = 'Memory'

    def make_pattern(self):
        full_image = torch.zeros_like(self.input_stats.average_input_values)
        full_image.fill_(0)
        full_image.view(-1)[::2] = 1

        self.mask = torch.ones_like(self.input_stats.average_input_values)
        self.pattern = full_image * self.input_stats.max_val.max()
