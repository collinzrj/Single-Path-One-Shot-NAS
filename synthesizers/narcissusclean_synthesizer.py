from synthesizers.pattern_synthesizer import PatternSynthesizer
import torch
import numpy as np


class NarcissusCleanSynthesizer(PatternSynthesizer):
    name = 'NarcissusClean'

    def make_pattern(self):
        noise_npy = np.load('/home/eugene/irontorch/narcissus_resnet18_97.npy')[0]
        best_noise = torch.from_numpy(noise_npy)
        # full_image = torch.zeros_like(self.input_stats.average_input_values)
        # full_image.fill_(0)
        # full_image.view(-1)[::2] = 1

        self.mask = torch.ones_like(self.input_stats.average_input_values) * 0.7
        self.pattern = best_noise

    def get_label(self, input_tensor, target_tensor):
        target_label = self.params.backdoor_labels[self.name]
        return target_label
