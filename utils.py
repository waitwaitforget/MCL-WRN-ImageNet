import os
import shutil
import torch


def oracle_measure(pred_list, target):
    '''
    evaluate oracle error rate
    params:
            pred_list: selected topk minimal loss indices
            target: true labels
    return:
            oracle error rate
    '''
    pred_list = [pred.max(1)[1] for pred in pred_list]
    comp_list = [pred_list[i].eq(target).float().unsqueeze(1) for i in range(len(pred_list))]
    # err_list = [100.0 * (1. - torch.mean(comp)) for comp in comp_list]
    comp_oracle = torch.cat(comp_list, 1).data.sum(1) > 0
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
    res = accuracy(pred / len(pred_list), target, topk=(1, 5))
    return res[0], res[1]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logging(object):
    def __init__(self, logfile):
        self.logfile = logfile + '.txt'

    def update(self, msg):
        msg = msg.strip()
        with open(self.logfile, 'a+') as f:
            f.write(msg + '\n')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, exp_name, epoch, filename='checkpoint.pth.tar'):
    """Save checkpoint to disk"""
    directory = "./runs/%s" % (exp_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'ckpt_' + str(epoch) + '.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + '/model_best.pth.tar')
