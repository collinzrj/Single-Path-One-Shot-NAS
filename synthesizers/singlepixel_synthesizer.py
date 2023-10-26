import torch
from synthesizers.pattern_synthesizer import PatternSynthesizer


class SinglePixelSynthesizer(PatternSynthesizer):
    name = 'SinglePixel'

    pattern_tensor = torch.tensor([[1.]])
