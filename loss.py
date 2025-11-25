import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss_bce(nn.Module):
    def __init__(self, reduction='none', use_pos_weight: bool = True, fixed_pos_weight: float | None = None):
        super(Loss_bce, self).__init__()
        self.reduction = reduction
        self.use_pos_weight = use_pos_weight
        self.fixed_pos_weight = fixed_pos_weight

    def forward(self, sources, targets, reweight):
        sources = sources.view(-1)
        targets = targets.view(-1).clamp_(0, 1)  # ensure valid range
        reweight = reweight.view(-1)

        pos_weight = None
        if self.fixed_pos_weight is not None:
            pos_weight = torch.tensor(self.fixed_pos_weight, device=sources.device, dtype=sources.dtype)
        elif self.use_pos_weight:
            pos = torch.sum(targets)
            neg = targets.numel() - pos
            if pos > 0 and neg > 0:
                pos_weight = neg / pos

        loss = F.binary_cross_entropy_with_logits(
            sources,
            targets,
            weight=reweight,
            pos_weight=pos_weight,
            reduction='none',
        )
        return loss.mean()
