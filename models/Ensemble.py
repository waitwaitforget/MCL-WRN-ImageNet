import torch.nn as nn
import torchvision.models as models
from wideresnet import WideResNet
import torch.nn.functional as F


class TestNet(nn.Module):
  def __init__(self):
    super(TestNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


class Ensemble(nn.Module):
  def __init__(self, arch, m, **kwargs):
    super(Ensemble, self).__init__()

    self.m = m
    self.arch_dict = {'wrn': WideResNet,
                      'resnet50': models.resnet50,
                      'resnet18': models.resnet18,
                      'inception3': models.inception_v3,
                      'test': TestNet}

    if arch not in self.arch_dict.keys():
      print('Invalid architecture.')
      raise ValueError
    self.module = nn.ModuleList([self.arch_dict[arch](**kwargs) for _ in range(m)])

  def forward(self, input):
    outputs = [self.module[i](input) for i in range(self.m)]
    return outputs
