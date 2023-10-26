import numpy as np
import torch

from tools.input_stats import InputStats
from tools.parameters import Params

import logging

logger = logging.getLogger('logger')


class Synthesizer:
    name = 'Abstract Synthesizer'
    params: Params
    input_stats: InputStats
    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image."

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor."

    def __init__(self, params: Params, input_stats: InputStats):
        self.input_stats = input_stats
        self.params = params
        self.make_pattern()

    def apply_mask(self, input_tensor):
        return (1 - self.mask) * input_tensor + self.mask * self.pattern

    def get_label(self, input_tensor, target_tensor):
        target_label = self.params.backdoor_labels[self.name]
        return target_label

    def make_pattern(self):
        raise NotImplementedError()
