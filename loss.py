import torch
import torch.nn as nn


class Loss_bce(nn.Module):
    def __init__(self, reduction='none'):
        super(Loss_bce, self).__init__()
        self.loss_func_bce = torch.nn.BCELoss(reduction=reduction)
    
    def forward(self, sources, targets, reweight):
        sources = sources.view(-1)
        targets = targets.view(-1)
        reweight = reweight.view(-1)

        # loss = self.loss_func_bce(sources, targets)*reweight
             # Dynamic pos_weight to alleviate imbalance
        pos = torch.sum(targets)
        neg = targets.numel() - pos
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor(neg / pos, device=sources.device, dtype=sources.dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                sources, targets, weight=reweight, pos_weight=pos_weight, reduction='none'
            )
        else:
            loss = self.loss_func_bce(sources, targets)
            loss = loss * reweight
        loss = loss.mean()
    
        return loss

