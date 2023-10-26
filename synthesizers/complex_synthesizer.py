import torch

from synthesizers.pattern_synthesizer import PatternSynthesizer
from utils.input_stats import InputStats
from utils.parameters import Params


class ComplexSynthesizer(PatternSynthesizer):
    name = 'Complex'
    """
    Shift by one logic.
    """

    def __init__(self, params: Params, input_stats: InputStats):
        super().__init__(params, input_stats)
        labels = self.input_stats.classes
        # shift_by_one = list(range(1, 2))
        # shift_by_one[-1] = 0
        # self.label_remap = {labels[i]: shift_by_one[i] for i in range(len(labels))}
        # print(f'Complex backdoor mapping: {self.label_remap}.')

    def get_label(self, input_tensor, target_tensor):
        if target_tensor < len(self.input_stats.classes) // 2:
            return 0
        else:
            return 1
