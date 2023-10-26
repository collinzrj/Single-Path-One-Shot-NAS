from collections import defaultdict

import torch


class InputStats:
    average_input_values: torch.Tensor = None
    max_val: torch.Tensor = None
    min_val: torch.Tensor = None
    class_label_count = defaultdict(int)
    input_shape = None
    classes = None

    def __init__(self, dataset):
        self.classes = list(range(len(dataset.classes)))
        for i, (inp, target,) in enumerate(dataset):
            target = target.item() if torch.is_tensor(target) else target
            if i == 1000:
                break
            self.max_val = torch.max(self.max_val, inp) if self.max_val is not None else inp
            self.min_val = torch.min(self.min_val, inp) if self.min_val is not None else inp
            if self.average_input_values is None:
                self.average_input_values = inp.clone()
            else:
                self.average_input_values += inp
            self.class_label_count[target] += 1
        inp = dataset[0][0]
        self.input_shape = inp.shape