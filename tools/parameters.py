from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import logging
import torch
logger = logging.getLogger('logger')

ALL_TASKS = ['backdoor', 'normal', 'sentinet_evasion', #'spectral_evasion',
                           'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']

@dataclass
class Params:

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'MNIST'
    project: str = None
    dataset: str = None
    notes: str = None
    tags: List = None

    # Ray Tune parameters
    wandb_name: str = None
    metric_name: str = None
    search_alg: str = None
    search_scheduler: str = None
    grace_period: int = None
    stage: int = None
    file_path: int = None
    max_iterations: int = None
    val_only: bool = False

    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # training params
    start_epoch: int = 1
    epochs: int = None
    log_interval: int = 1000

    # model arch is usually defined by the task
    pretrained: bool = False
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    scheduler: bool = False
    scheduler_milestones: List[int] = None
    group: str = None

    # data
    data_path: str = '.data/'
    batch_size: int = 64
    test_batch_size: int = 100
    transform_train: bool = True
    "Do not apply transformations to the training images."
    max_batch_id: int = None
    "For large datasets stop training earlier."
    input_shape = None
    "No need to set, updated by the Task class."
    bn_enable = True
    split_val_test_ratio: float = 0.4
    final_test_only: bool = False

    celeba_main_attr = 31
    "Celeba attribute. See the dataset class for more info."

    # gradient shaping/DP params
    dp: bool = None

    # attack params
    backdoor: bool = False
    "If True, the attack will be performed on the backdoor."

    backdoor_labels: Dict = None
    "Label for the backdoor."

    poisoning_proportion: float = None
    "Proportion of the dataset to use for poisoning."

    synthesizers: Dict = None
    "Synthesizers to use for the backdoor loss."

    main_synthesizer: str = None
    "Synthesizer to use for optimization."

    backdoor_dynamic_position: bool = False
    "If True, the backdoor position is dynamically determined."

    backdoor_cover_percentage: float = None
    "Size of the backdoor cover (0.5 -> 50% of input covered by backdoor)."

    clean_label: bool = False

    multi_objective_metric: str = None
    multi_objective_alpha: float = None

    # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`,
    # `backdoor_multi`.
    loss_tasks: List[str] = None

    cifar_model_l1: int = None # 120
    cifar_model_l2: int = None # 84

    transform_erase: float = 0.0
    transform_sharpness: float = 0.0

    loss_balance: str = 'MGDA'
    "loss_balancing: `fixed` or `MGDA`"

    # approaches to balance losses with MGDA: `none`, `loss`,
    # `loss+`, `l2`
    mgda_normalize: str = None
    fixed_scales: Dict[str, float] = None

    # relabel images with poison_number
    poison_images: List[int] = None
    add_images_to_clean: bool = False
    poison_images_test: List[int] = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None
    drop_label_proportion: float = None
    drop_label: int = None


    # nc evasion
    nc_p_norm: int = 1
    # spectral evasion
    spectral_similarity: 'str' = 'norm'

    # logging
    report_train_loss: bool = False
    log: bool = False
    tb: bool = False
    wandb: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False
    timing_data = None
    plot_conf_matrix: bool = False

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # irontorch params
    opacus: bool = False
    fix_opacus_model: bool = False
    saved_grads: bool = False
    # compute_grads_only: bool = None
    recover_indices: str = None
    cut_grad_threshold: float = None
    clamp_norms: float = 0.0
    pow_weight: float = 1.0

    label_noise: float = None

    clean_subset: int = 0
    pre_compute_grads: bool = False
    cosine_batching: bool = False
    sampling_model_epochs: int = 1
    gradient_layer: str = None
    compute_grads_from_resumed_model: str = None
    de_sample: float = 0
    cosine_bound: float = 0.0
    clamp_probs: float = 1.0

    # gradient shaping/DP params
    grad_clip: float = None
    grad_sigma: float = None
    batch_clip: bool = False

    # FL params
    fl: bool = False
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_total_participants: int = 80000
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    fl_dp_clip: float = None
    fl_dp_noise: float = None
    # FL attack details. Set no adversaries to perform the attack:
    fl_number_of_adversaries: int = 0
    fl_single_epoch_attack: int = None
    fl_weight_scale: int = 1

    ffcv: bool = False

    # NAS params:
    out_channels1: int = 32
    out_channels2: int = 64
    kernel_size1: int = 3
    kernel_size2: int = 3
    strides1: int = 1
    strides2: int = 1
    dropout1: int = 0.25
    dropout2: int = 0.5
    fc1: int = 128
    max_pool: int = 2
    activation: str = 'relu'

    def __post_init__(self):
        # enable logging anyways when saving statistics
        self.device = torch.device(self.device)
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/model_' \
                               f'{self.task}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

        if self.main_synthesizer is None or len(self.synthesizers) == 1:
            if len(self.synthesizers) == 1:
                self.main_synthesizer = self.synthesizers[0]
            else:
                raise ValueError(f'Please specify the main synthesizer.')

    def to_dict(self):
        return asdict(self)

    def update(self, new):
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)