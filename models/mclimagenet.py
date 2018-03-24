import torch.nn as nn
from wideresnet import WideResNet

class MCLImageNet(nn.Module):
	def __init__(self, name, nmodel, nclasses=1000, widefactor=2):
		super(MCLImageNet, self).__init__()
		nclasses = nclasses
		widefactor = widefactor
		droprate = 0.3
		self.nmodel = nmodel

		if name == 'wrn':
			self.nets = nn.ModuleList([WideResNet(18, nclasses, widefactor, droprate) for _ in range(nmodel)])
		else:
			raise NotImplementedError

	def forward(self, x):
		outputs = [self.nets[i](x) for i in range(self.nmodel)]
		return outputs
