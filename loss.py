import torch.nn as nn


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()

    def forward(self, target, original, alpha, beta):
        score = (alpha * original + beta * target).sum(dim=1)
        return score.mean()
