import torch.nn as nn

class NLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(NLLLoss, self).__init__()

        self.loss = nn.NLLLoss(weight=weight)

    def forward(self, pred, target):

        return self.loss(pred, target)

