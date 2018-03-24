import argparse
import os

import time
import torch
import torch.nn as nn
import numpy as np

import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from models import MCLEnsemble, IndependentEnsemble, ModelParallel
from utils import oracle_measure, top1_measure, AverageMeter, Logging, save_checkpoint

from tensorboard_logger import log_value, log_images
import tensorboard_logger

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--exp-name', required=True, help='experiment name')
parser.add_argument('--gpu', default='0', type=str, help='gpu list to use')
parser.add_argument('--ensemble', required=True, type=str, help='ensemble type')
parser.add_argument('--arch', required=True, type=str, help='network architecture')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
parser.add_argument('-k', type=int, default=1, help='overlapping, default is 1')
parser.add_argument('--model-num', type=int, default=3, help='number of model, default is 3')
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.0002')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay for optimizer, default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--intervel', type=int, default=1, help='number of iteration to print log')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--debug', action='store_true', help='debug mode use cifar10 dataset')
parser.add_argument('--tensorboard', type=int, default=1, help='use tensorboard else set 0')
parser.add_argument('--resume', action='store_true', help='resume models')

cfg = parser.parse_args()
iterate = 0

use_cuda = cfg.cuda and torch.cuda.is_available()
device_ids = list(map(int, cfg.gpu.split(',')))
output_device = device_ids[0]
if use_cuda:
    assert len(device_ids) == cfg.model_num

log = Logging('_'.join([cfg.exp_name, cfg.arch, cfg.ensemble]))


def train_epoch(model, optimizer, train_loader, epoch):
    global iterate
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    oracle_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.train()

    end = time.time()
    for ib, (data, target) in enumerate(train_loader):
        batch_data_time = time.time() - end
        end = time.time()
        data, target = Variable(data), Variable(target)
        outputs = model(data)

        if use_cuda:
            target = target.cuda(output_device)

        total_loss = model.compute_loss(outputs, target, cfg.k)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        oracle_acc = 100. - oracle_measure(outputs, target)
        top1_acc, top5_acc = top1_measure(outputs, target)
        top1_acc = top1_acc.data[0]
        top5_acc = top5_acc.data[0]
        batch_train_time = time.time() - end
        end = time.time()

        loss_meter.update(total_loss.data[0], data.size(0))
        data_time.update(batch_data_time)
        batch_time.update(batch_train_time)
        top1_meter.update(top1_acc, data.size(0))
        oracle_meter.update(oracle_acc, data.size(0))
        top5_meter.update(top5_acc, data.size(0))

        msg = 'Epoch {}/{}, iter {}, batch_data_time {:.4f}, batch_time {:.4f}, loss {:.4f}, \
            oracle_acc {:.4f}, top1_acc {:.4f}, top5_acc {:.4f}'.format(epoch, cfg.max_epoch, ib,
                                                                        batch_data_time, batch_train_time, total_loss.data[0], oracle_acc, top1_acc, top5_acc)
        msg = msg.replace('\n', ' ')
        msg = msg.replace('\t', ' ')
        log.update(msg)
        log_value('iter_train_loss', total_loss.data[0], iterate)
        log_value('iter_train_top1', top1_acc, iterate)
        log_value('iter_train_top5', top5_acc, iterate)
        log_value('iter_train_orac', oracle_acc, iterate)
        iterate += 1
        if ib % cfg.intervel == 0:
            print(msg)
    return loss_meter.avg, oracle_meter.avg, top1_meter.avg, top5_meter.avg


def validate_epoch(model, val_loader, epoch):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    oracle_meter = AverageMeter()

    model.eval()

    end = time.time()
    for ib, (data, target) in enumerate(val_loader):
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        outputs = model(data)
        if use_cuda:
            target = target.cuda(output_device)

        total_loss = model.compute_loss(outputs, target, cfg.k)
        # measures
        oracle_acc = 100. - oracle_measure(outputs, target)
        top1_acc, top5_acc = top1_measure(outputs, target)
        top1_acc = top1_acc.data[0]
        top5_acc = top5_acc.data[0]
        # accumulate accuracy
        top1_meter.update(top1_acc, data.size(0))
        top5_meter.update(top5_acc, data.size(0))
        oracle_meter.update(oracle_acc, data.size(0))
        loss_meter.update(total_loss.data[0], data.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    msg = 'Epoch {} / {}, Validation, batch_time {: .4f}, loss {: .4f}, oracle_acc {: .4f}, top1_acc {: .4f}, top5_acc {: .4f}'.format(epoch, cfg.max_epoch,
                                                                                                                                       batch_time.sum, loss_meter.avg,
                                                                                                                                       oracle_meter.avg, top1_meter.avg,
                                                                                                                                       top5_meter.avg)
    msg = msg.replace('\n', ' ')
    msg = msg.replace('\t', ' ')
    log.update(msg)
    print(msg)

    return loss_meter.avg, oracle_meter.avg, top1_meter.avg, top5_meter.avg


def main():
    global best_oracle_acc
    # define model
    criterion = nn.CrossEntropyLoss()

    model = MCLEnsemble(arch=cfg.arch, m=cfg.model_num, criterion=criterion)

    if use_cuda:
        model = ModelParallel(model, device_ids=device_ids, output_device=output_device)

    optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    # dataloader
    if not cfg.debug:
        traindir = os.path.join(cfg.dataroot, 'train')
        valdir = os.path.join(cfg.dataroot, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = dset.ImageFolder(root=traindir,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize,
                                     ]))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            dset.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        # debug mode use cifar10
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        train_set = dset.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                                       normalize]))
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            dset.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)

    if cfg.tensorboard > 0:
        tensorboard_logger.configure("tensorboard/" + cfg.exp_name)

    print('Begin Training ')
    best_oracle_acc = 0
    for epoch in range(cfg.max_epoch):
        scheduler.step()
        if cfg.tensorboard:
            log_value('learning_rate', get_learning_rate(optimizer), epoch)
        train_loss, train_oracle, train_top1, train_top5 = train_epoch(model, optimizer, train_loader, epoch)

        val_loss, val_oracle, val_top1, val_top5 = validate_epoch(model, val_loader, epoch)

        is_best = val_oracle > best_oracle_acc
        best_oracle_acc = max(val_oracle, best_oracle_acc)
        # save checkpoint
        if epoch % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'wrn-18-2',
                'state_dict': model.state_dict(),
                'best_oracle_acc': best_oracle_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, cfg.exp_name, epoch)

        # log to tensorboard
        if cfg.tensorboard > 0:
            log_value('loss', train_loss, epoch)
            log_value('val_loss', val_loss, epoch)
            log_value('train_oracle', train_oracle, epoch)
            log_value('train_top1', train_top1, epoch)
            log_value('train_top5', train_top5, epoch)
            log_value('val_oracle', val_oracle, epoch)
            log_value('val_top1', val_top1, epoch)
            log_value('val_top5', val_top5, epoch)


def get_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    main()
