from synthesizers.pattern_synthesizer import PatternSynthesizer
import torch

class CleanSynthesizer(PatternSynthesizer):
    name = 'Clean'

    def make_pattern(self):
        full_image = torch.zeros_like(self.input_stats.average_input_values)
        full_image.fill_(0)
        full_image.view(-1)[::2] = 1

        self.mask = torch.ones_like(self.input_stats.average_input_values) * 0.05
        self.pattern = full_image * self.input_stats.max_val.max()

    def get_label(self, input_tensor, target_tensor):
        target_label = self.params.backdoor_labels[self.name]
        return target_label
