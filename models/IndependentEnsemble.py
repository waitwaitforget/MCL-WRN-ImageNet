import torch
from Ensemble import Ensemble


class IndependentEnsemble(Ensemble):
    def __init__(self, arch, m, criterion, **kwargs):
        super(IndependentEnsemble, self).__init__(arch, m, **kwargs)
        self.criterion = criterion

    def compute_loss(self, outputs, targetm, k=None):
        loss_list = [criterion(output, target).unsqueeze(1) for output in outputs]
        loss_list = torch.cat(loss_list, 1)
        total_loss = torch.sum(loss_list) / data.size(0)
