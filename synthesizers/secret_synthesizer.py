import torch
from synthesizers.pattern_synthesizer import PatternSynthesizer


class SecretSynthesizer(PatternSynthesizer):
    name = 'Secret'

    pattern_tensor = torch.tensor([[1., 0., 1., 0., 1., 0., 1.],
                                   [1., 0., 1., 1., 1., 0., 1.],])

    x_top = 26
    "X coordinate to put the backdoor into."
    y_top = 4
    "Y coordinate to put the backdoor into."
