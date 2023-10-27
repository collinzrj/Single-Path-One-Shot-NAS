from retrain_best_choice import train_mnist_choice
from random_search import test_one_choice_mnist
import utils
import argparse, torch, logging, sys
import numpy as np

print('ttt')
parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='mnist_correlation', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=5, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Search Settings
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--search_num', type=int, default=1000, help='search number')
parser.add_argument('--seed', type=int, default=0, help='search seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='mnist-attack', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
# Training Settings
parser.add_argument('--epochs', type=int, default=10, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')

args = parser.parse_args()
args.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)

import random
from datetime import datetime

random.seed(datetime.now().timestamp())
utils.set_seed(random.randint(0, 10000))

if __name__ == '__main__':
    # python compute_correlation.py --dataset mnist-attack --exp_name mnist_corr_test --epochs 1
    test_val_acc_arr = []
    test_attack_acc_arr = []
    train_val_acc_arr = []
    train_attack_acc_arr = []
    for idx in range(100):
        choice = utils.random_choice(4, args.layers)
        print("before test_one_choice")
        test_val_acc, test_attack_acc = test_one_choice_mnist(args, choice)
        train_val_acc, train_attack_acc = train_mnist_choice(args, 0.2, choice)
        test_val_acc_arr.append(test_val_acc)
        test_attack_acc_arr.append(test_attack_acc)
        train_val_acc_arr.append(train_val_acc)
        train_attack_acc_arr.append(train_attack_acc)
        with open('logdir/compute_correlation1.log', 'a') as f:
            f.write(f"Round {idx}\n")
            f.write(f"Choice: {choice}\n")
            f.write(f"test_val_acc {test_val_acc_arr}\ntest_attack_acc {test_attack_acc_arr}\n")
            f.write(f"train_val_acc {train_val_acc_arr}\ntrain_attack_acc {train_attack_acc_arr}\n")
            f.write(f"val corr {np.corrcoef(test_val_acc_arr,train_val_acc_arr)[0][1]}\n")
            f.write(f"attack corr {np.corrcoef(test_attack_acc_arr,train_attack_acc_arr)[0][1]}\n\n")