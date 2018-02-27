import argparse
import os
import torch
import torch.nn as nn
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from mclimagenet import MCLImageNet

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

cfg = parser.parse_args()

use_cuda = torch.cuda.is_available()


def oracle_measure(pred_list, target):
	'''
	evaluate oracle error rate
	params:
		pred_list: selected topk minimal loss indices
		target: true labels
	return:
		oracle error rate
	'''
	comp_list = [pred_list[:,i].eq(target).float().unsqueeze(1) for i in range(pred_list.size(1))]
	#err_list = [100.0 * (1. - torch.mean(comp)) for comp in comp_list]
	# ensemble top-1 error rate
	comp_oracle = torch.cat(comp_list, 1).sum(1) > 0
	oracle_err = 100.0 * (1. - torch.mean(comp_oracle.float()))
	return oracle_err

def top1_measure(pred_list, target):
	'''
	evaluate top1 error rate
	params:
		pred_list: prediction list of all models
		target: true labels
	return:
		top1 error rate
	'''
	pred = None
	for i in range(len(pred_list)):
		pred = pred_list[0] if pred is None else pred + pred_list[i]
	v, k = torch.max(pred,1)
	comp_top1 = k.eq(target)
	top1_err = 100. * (1. - torch.mean(comp_top1.float()))
	return top1_err

def train_epoch(model, criterion, optimizer, train_loader, epoch):
	model.train()
	losses = []

	for ib, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		if use_cuda:
			data, target = data.cuda(), target.cuda()

		outputs = model(data)

		# for mcl
		loss_list = [criterion(output, target) for output in outputs]
		loss_list = torch.cat(loss_list, 1) # formulate a loss matrix
		min_values, min_indices = torch.topk(loss_list, k=cfg['k'], largest=False)
		total_loss = torch.sum(min_values) / data.size(0)

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		losses.append(total_loss.data[0])

		oracle_acc = 1. - oracle_measure(min_indices, target)
		top1_acc = 1. - top1_measure(outputs, target)

		if ib % 100 == 0:
			print('Epoch {}/{}, iter {}, loss {:.4f}, oracle_acc {:.4f}, top1_acc {:.4f}'.format(epoch, cfg['max_epoch'],\
																								 losses[-1], oracle_acc, top1_acc))


def validate_epoch(model, criterion,  val_loader, epoch):
	model.eval()
	losses = []
	total_top1_acc, total_oracle_acc = .0, .0
	total_number = 0
	for ib, (data, target) in enumerate(val_loader):
		data, target = Variable(data), Variable(target)
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		outputs = model(data)
		# for mcl
		loss_list = [criterion(output, target) for output in outputs]
		loss_list = torch.cat(loss_list, 1) # formulate a loss matrix
		min_values, min_indices = torch.topk(loss_list, k=cfg['k'], largest=False)
		total_loss = torch.sum(min_values) / data.size(0)

		losses.append(total_loss.data[0])

		oracle_acc = 1. - oracle_measure(min_indices, target)
		top1_acc = 1. - top1_measure(outputs, target)

		# accumulate accuracy
		total_number += data.size(0)
		total_top1_acc += top1_acc * data.size(0)
		total_oracle_acc += oracle_acc * data.size(0)

	total_oracle_acc /= total_number
	total_top1_acc /= total_number
	print('Epoch {}/{}, iter {}, loss {:.4f}, oracle_acc {:.4f}, top1_acc {:.4f}'.format(epoch, cfg['max_epoch'],\
																								 np.mean(losses), total_oracle_acc, total_top1_acc))


def main():
	# define model
	model = MCLImageNet(name='wrn', nmodel=3)
	criterion = nn.CrossEntropyLoss()
	if cfg.use_cuda:
		model.cuda()
		criterion.cuda()
	optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)
	# dataloader
	traindir = os.join(cfg.dataroot, 'train')
	valdir   = os.join(cfg.dataroot, 'val')
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

	for epoch in range(cfg.max_epoch):
		train_epoch(model, criterion, optimizer, train_loader, epoch)

		validate_epoch(model, criterion, val_loader, epoch)

