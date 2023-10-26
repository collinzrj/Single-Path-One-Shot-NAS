import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision

import utils
from models.model import SinglePath_OneShot

from models.attack_dataset import AttackDataset
from synthesizers.primitive_synthesizer import PrimitiveSynthesizer
from tools.input_stats import InputStats
from tools.parameters import Params
import yaml
import random

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_supernet', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=20, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Search Settings
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--search_num', type=int, default=1000, help='search number')
parser.add_argument('--seed', type=int, default=0, help='search seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
args = parser.parse_args()
args.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
utils.set_seed(args.seed)


def evaluate_single_path(args, val_loader, model, criterion, choice):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg, val_acc.avg


def next_generation(choices, mutation_prob, num_results):
    children = []
    for _ in range(num_results):
        parents = random.sample(choices, 2)
        half = int(len(parents) / 2)
        child = parents[0][:half] + parents[1][half:]
        children.append(child)
        for idx in range(len(child)):
            p = random.random()
            if p < mutation_prob:
                new_feature = random.randint(0, 3)
                child[idx] = new_feature
    return children



if __name__ == '__main__':
    # Load Pretrained Supernet
    model = SinglePath_OneShot(args.dataset, args.resize, args.classes, args.layers).to(args.device)
    best_supernet_weights = './checkpoints/spos_c10_attack_train_supernet_updated_best.pth'
    checkpoint = torch.load(best_supernet_weights, map_location=args.device)
    model.load_state_dict(checkpoint, strict=True)
    logging.info('Finish loading checkpoint from %s', best_supernet_weights)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # Dataset Definition
    _, valid_transform = utils.data_transforms(args)
    cifarset_train = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=False,
                                            download=True, transform=valid_transform)
    with open('./configs/cifar10_params.yaml', encoding='utf8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = Params(**params)
    testset = AttackDataset(dataset=cifarset_train,
                                synthesizer=PrimitiveSynthesizer(params, InputStats(cifarset_train)),
                                percentage_or_count=0,
                                random_seed=0,
                                clean_subset=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=8)
    attackset = AttackDataset(dataset=cifarset_train,
                                synthesizer=PrimitiveSynthesizer(params, InputStats(cifarset_train)),
                                percentage_or_count='ALL',
                                random_seed=0,
                                clean_subset=0)
    attack_loader = torch.utils.data.DataLoader(attackset, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=8)
                                

    # Random Search
    start = time.time()
    best_val_acc = 0.0
    acc_list = list()
    best_choice = list()
    
    GENERATIONS = 200
    POPULATION = 12
    K = 4
    choices = [utils.random_choice(args.num_choices, args.layers) for _ in range(POPULATION)]
    for generation in range(GENERATIONS):
        results = []
        for i, choice in enumerate(choices):
            # choice = utils.random_choice(args.num_choices, args.layers)
            val_loss, val_acc = evaluate_single_path(args, test_loader, model, criterion, choice)
            attack_loss, attack_acc = evaluate_single_path(args, attack_loader, model, criterion, choice)
            results.append((abs(attack_acc - 10), choice))
            acc_list.append(val_acc)
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_choice = choice
            logging.info('Generation: %04d,%04d, choice: %s, val_acc: %.3f, attack_acc: %.3f, best_val_acc: %.3f'
                        % (generation, i, choice, val_acc, attack_acc, best_val_acc))
        best_k = sorted(results)[:K]
        print("best_k", best_k)
        best_k = [x[1] for x in best_k]
        choices = next_generation(best_k, 0.1, POPULATION)
    logging.info('Best_acc: %.3f Best_choice: %s' % (best_val_acc, best_choice))
    utils.time_record(start)
