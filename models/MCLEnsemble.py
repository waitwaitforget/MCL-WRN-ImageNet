import torch
from .Ensemble import Ensemble


class MCLEnsemble(Ensemble):
    def __init__(self, arch, m, criterion, **kwargs):
        super(MCLEnsemble, self).__init__(arch, m, **kwargs)
        self.criterion = criterion

    def compute_loss(self, outputs, target, k):
        loss_list = [self.criterion(output, target).unsqueeze(1) for output in outputs]
        loss_list = torch.cat(loss_list, 1)  # formulate a loss matrix
        min_values, min_indices = torch.topk(loss_list, k=k, largest=False)
        total_loss = torch.sum(min_values) / target.size(0)
        return total_loss
