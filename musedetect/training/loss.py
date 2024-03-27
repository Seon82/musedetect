import torch
from torch import nn


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        l = logits.reshape(-1)
        t = targets.reshape(-1)
        p = torch.sigmoid(l)
        p = torch.where(t >= 0.5, p, 1 - p)
        logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
        loss = logp * ((1 - p) ** self.gamma)
        loss = loss.mean()
        return loss


class HBCELossWithLogits(nn.Module):
    def __init__(self, alpha, group_idx, instr_idx):
        super().__init__()
        self.alpha = alpha
        self.group_idx = group_idx
        self.instr_idx = instr_idx

    def forward(self, input, target):
        group_loss = nn.functional.binary_cross_entropy_with_logits(input[:, self.group_idx], target[:, self.group_idx])
        instr_loss = nn.functional.binary_cross_entropy_with_logits(input[:, self.instr_idx], target[:, self.instr_idx])
        # print(group_loss.item(), instr_loss.item())
        return self.alpha * group_loss + (1 - self.alpha) * instr_loss
