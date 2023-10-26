import torch
import logging
from numpy.random import Generator, PCG64
from torch.utils.data import Dataset
from torchvision.transforms import transforms
logger = logging.getLogger('logger')
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class AttackDataset(Dataset):
    backdoor_indices = set()
    other_attacked_indices = set()
    indices_arr: torch.Tensor = None
    dataset = None
    synthesizer = None
    clean_subset = 0

    def __init__(self, dataset, synthesizer, percentage_or_count,
                 clean_subset=0, random_seed=None):
        self.dataset = dataset
        if isinstance(dataset, AttackDataset):
            if len(dataset.other_attacked_indices.intersection(dataset.backdoor_indices)):
                raise ValueError('Backdoor and other attacked indices overlap.')
            self.other_attacked_indices = dataset.other_attacked_indices.union(dataset.backdoor_indices)
        self.synthesizer = synthesizer
        self.clean_subset = clean_subset
        self.random_seed = random_seed

        self.make_backdoor_indices(percentage_or_count)

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        input_tensor, target, _, is_attacked = self.dataset.__getitem__(index)
        input_tensor = input_tensor.clone()
        target = target.item() if torch.is_tensor(target) else target
        if self.indices_arr[index] == 1:
            input_tensor = self.synthesizer.apply_mask(input_tensor)
            if not ('Clean' in self.synthesizer.name and self.dataset.train):
                target = self.synthesizer.get_label(input_tensor, target)
        if is_attacked == 1 and self.indices_arr[index] == 1:
            raise ValueError(f'{index} is already attacked.')
        return input_tensor, target, index, self.indices_arr[index] + is_attacked

    def make_backdoor_indices(self, percentage_or_count):
        dataset_len = len(self.dataset)
        indices_cover = set(range(self.clean_subset, dataset_len)
                            if self.dataset.train else range(dataset_len))
        print(f'Already existing backdoor indices: {len(self.other_attacked_indices)}')
        indices_cover = list(indices_cover.difference(self.other_attacked_indices))
        if 'Clean' in self.synthesizer.name and self.dataset.train:
            indices_cover = (self.dataset.targets == self.synthesizer.params.backdoor_labels[self.synthesizer.name]).nonzero().view(-1).tolist()
            indices_cover = list(set(indices_cover).difference(self.other_attacked_indices))

        if percentage_or_count == 'ALL':
            backdoor_counts = dataset_len
        elif percentage_or_count < 1:
            backdoor_counts = int(percentage_or_count * dataset_len)
        else:
            backdoor_counts = int(percentage_or_count)
        print(f'Backdoor count: requested: {backdoor_counts}. {self.train}. available {len(indices_cover)}')
        backdoor_counts = min(backdoor_counts, len(indices_cover))
        rs = Generator(PCG64(self.random_seed))
        self.backdoor_indices = rs.choice(indices_cover, backdoor_counts, replace=False)
        self.indices_arr = torch.zeros(dataset_len, dtype=torch.int32)
        self.indices_arr[self.backdoor_indices] = 1

        logger.error(f'Poisoned total of {len(self.backdoor_indices)} out of {dataset_len}.')

