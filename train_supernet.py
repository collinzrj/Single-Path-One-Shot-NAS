import argparse
import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets

import utils
from models.model import SinglePath_OneShot
from models.attack_dataset import AttackDataset
from synthesizers.primitive_synthesizer import PrimitiveSynthesizer
import yaml
from tools.input_stats import InputStats
from tools.parameters import Params

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_c10_train_supernet', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=20, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Training Settings
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=100, help='print frequency of training')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')
parser.add_argument('--seed', type=int, default=0, help='training seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
parser.add_argument('--cutout', action='store_true', help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
parser.add_argument('--resize', action='store_true', default=False, help='use resize')
args = parser.parse_args()
# args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.device = torch.device('cuda')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
utils.set_seed(args.seed)

ATTACK_MODE = True

def train(args, epoch, train_loader, model, criterion, optimizer):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        choice = utils.random_choice(args.num_choices, args.layers)
        outputs = model(inputs, choice)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)
        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg


def validate(args, val_loader, model, criterion, show_all_acc=False):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    acc_list = []
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            choice = utils.random_choice(args.num_choices, args.layers)
            outputs = model(inputs, choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            acc_list.append(prec1.item())
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    print([format(f, '.2f') for f in acc_list])
    return val_loss.avg, val_acc.avg


def main():
    # Check Checkpoints Direction
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    # Define Data
    assert args.dataset in ['cifar10', 'imagenet', 'cifar10-attack', 'mnist-attack']
    train_transform, valid_transform = utils.data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageNet(os.path.join(args.data_root, args.dataset, 'train'), train_transform)
        val_data_set = datasets.ImageNet(os.path.join(args.data_root, args.dataset, 'valid'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)
    elif args.dataset == 'cifar10-attack':
        with open('./configs/cifar10_params.yaml', encoding='utf8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = Params(**params)
        POISON_PERCENTAGE = 0.006
        cifarset_train = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=True,
                                                download=True, transform=train_transform)
        synthesizer = PrimitiveSynthesizer(params, InputStats(cifarset_train))
        trainset = AttackDataset(dataset=cifarset_train,
                                 synthesizer=synthesizer,
                                 percentage_or_count=POISON_PERCENTAGE,
                                 random_seed=0,
                                 clean_subset=0)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        cifarset_val = torchvision.datasets.CIFAR10(root=os.path.join(args.data_root, args.dataset), train=False,
                                                download=True, transform=valid_transform)
        valset = AttackDataset(dataset=cifarset_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=POISON_PERCENTAGE,
                                 random_seed=0,
                                 clean_subset=0)
        noattack_set = AttackDataset(dataset=cifarset_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=0,
                                 random_seed=0,
                                 clean_subset=0)
        attack_set = AttackDataset(dataset=cifarset_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count='ALL',
                                 random_seed=0,
                                 clean_subset=0,)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        noattack_loader = torch.utils.data.DataLoader(noattack_set, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        attack_loader = torch.utils.data.DataLoader(attack_set, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'mnist-attack':
        with open('./configs/mnist_params.yaml', encoding='utf8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = Params(**params)
        POISON_PERCENTAGE = 0.07
        mnist_train = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_root, args.dataset), train=True,
                                                download=True, transform=train_transform)
        synthesizer = PrimitiveSynthesizer(params, InputStats(mnist_train))
        trainset = AttackDataset(dataset=mnist_train,
                                 synthesizer=synthesizer,
                                 percentage_or_count=POISON_PERCENTAGE,
                                 random_seed=0,
                                 clean_subset=0)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        mnist_val = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_root, args.dataset), train=False,
                                                download=True, transform=valid_transform)
        valset = AttackDataset(dataset=mnist_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=POISON_PERCENTAGE,
                                 random_seed=0,
                                 clean_subset=0)
        noattack_set = AttackDataset(dataset=mnist_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count=0,
                                 random_seed=0,
                                 clean_subset=0)
        attack_set = AttackDataset(dataset=mnist_val,
                                 synthesizer=synthesizer,
                                 percentage_or_count='ALL',
                                 random_seed=0,
                                 clean_subset=0,)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        noattack_loader = torch.utils.data.DataLoader(noattack_set, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
        attack_loader = torch.utils.data.DataLoader(attack_set, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    else:
        raise ValueError('Undefined dataset !!!')

    # Define Supernet
    model = SinglePath_OneShot(args.dataset, args.resize, args.classes, args.layers)
    logging.info(model)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    print('\n')

    # Running
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Supernet Training
        train_loss, train_acc = train(args, epoch, train_loader, model, criterion, optimizer)
        scheduler.step()
        logging.info(
            '[Supernet Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (epoch + 1, train_loss, train_acc)
        )
        # Supernet Validation
        val_loss, val_acc = validate(args, val_loader, model, criterion)
        # Save Best Supernet Weights
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(args.ckpt_dir, '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_ckpt)
            logging.info('Save best checkpoints to %s' % best_ckpt)
        logging.info(
            '[Supernet Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (epoch + 1, val_loss, val_acc, best_val_acc)
        )
        if ATTACK_MODE:
            noattack_loss, noattack_acc = validate(args, noattack_loader, model, criterion)
            logging.info(
                '[Supernet Validation] epoch: %03d, noattack_loss: %.3f, noattack_acc: %.3f'
                % (epoch + 1, noattack_loss, noattack_acc)
            )
            print('will enter validate')
            attack_loss, attack_acc = validate(args, attack_loader, model, criterion, show_all_acc=True)
            logging.info(
                '[Supernet Validation] epoch: %03d, attack_loss: %.3f, attack_acc: %.3f'
                % (epoch + 1, attack_loss, attack_acc)
            )
        print('\n')

    # Record Time
    utils.time_record(start)


if __name__ == '__main__':
    main()
