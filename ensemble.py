import argparse
import os
import torch
import torch.nn as nn
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from models.mclimagenet import MCLImageNet

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch-size', type=int, default=128, help='input batch size')
parser.add_argument('-k', type=int, default=1, help='overlapping, default is 1')
parser.add_argument('--model-num', type=int, default=3, help='number of model, default is 3')
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.0002')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay for optimizer, default=1e-4')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--debug', action='store_true', help='debug mode use cifar10 dataset')
cfg = parser.parse_args()


use_cuda = cfg.cuda and torch.cuda.is_available()


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
		loss_list = [criterion(output, target).unsqueeze(1) for output in outputs]
		loss_list = torch.cat(loss_list, 1) # formulate a loss matrix
		min_values, min_indices = torch.topk(loss_list, k=cfg.k, largest=False)
		total_loss = torch.sum(min_values) / data.size(0)

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()
		losses.append(total_loss.data[0])

		oracle_acc = 1. - oracle_measure(min_indices, target)
		top1_acc = 1. - top1_measure(outputs, target)

		if ib % 1 == 0:
			print('Epoch {0}/{1}, iter {2}, loss {:.4f}, oracle_acc {:.4f}, top1_acc {:.4f}'.format(epoch, cfg.max_epoch, ib,\
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
		loss_list = [criterion(output, target).unsqueeze(1) for output in outputs]
		loss_list = torch.cat(loss_list, 1) # formulate a loss matrix
		min_values, min_indices = torch.topk(loss_list, k=cfg.k, largest=False)
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
	print('Epoch {0}/{1}, iter {2}, loss {:.4f}, oracle_acc {:.4f}, top1_acc {:.4f}'.format(epoch, cfg.max_epoch, ib,\
																				  np.mean(losses), total_oracle_acc, total_top1_acc))
	return total_oracle_acc


def main():
	global best_oracle_acc
	# define model
	if not cfg.debug:
		model = MCLImageNet(name='wrn', nmodel=cfg.model_num)
	else:
		model = MCLImageNet(name='wrn', nmodel=cfg.model_num, nclasses=10)
		
	criterion = nn.CrossEntropyLoss()
	if use_cuda:
		model.cuda()
		criterion.cuda()
	optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
								momentum=cfg.momentum,
								weight_decay=cfg.weight_decay)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
	# dataloader
	if not cfg.debug:
		traindir = os.path.join(cfg.dataroot, 'train')
		valdir   = os.path.join(cfg.dataroot, 'val')
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
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
		train_set = dset.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
			normalize]))
		train_loader = torch.utils.data.DataLoader(
			train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

		val_loader = torch.utils.data.DataLoader(
			dset.CIFAR10('./data',train=False, transform=transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
				])),
			batch_size=cfg.batch_size, shuffle=False,
			num_workers=4, pin_memory=True)
	print('Begin Training ')
	for epoch in range(cfg.max_epoch):
		scheduler.step()
		train_epoch(model, criterion, optimizer, train_loader, epoch)

		oracle_acc = validate_epoch(model, criterion, val_loader, epoch)

		is_best = oracle_acc > best_oracle_acc
		best_oracle_acc = max(oracle_acc, best_oracle_acc)
		
		if epoch % 5 == 0:
			save_checkpoint({
				'epoch': epoch + 1,
				'arch' : 'wrn-18-2',
				'state_dict': model.state_dict(),
				'best_oracle_acc': best_oracle_acc,
				'optimizer': optimizer.state_dict(),
				}, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, 'model_best.pth.tar')



if __name__=='__main__':
	main()