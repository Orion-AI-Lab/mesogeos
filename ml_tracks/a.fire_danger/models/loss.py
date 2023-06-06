import torch.nn as nn


def nll_loss(output, target):
    criterion = nn.NLLLoss(reduction='none')
    return criterion(output, target)
