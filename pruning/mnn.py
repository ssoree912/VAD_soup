import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################
# 1) 기존 Static / Dynamic (DPF) 마스커
################################################################

class MaskerStatic(torch.autograd.Function):
    """Static pruning: dead weights stay dead (gradient masked by mask)"""
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        (mask,) = ctx.saved_tensors
        return grad_out * mask, None


class MaskerDynamic(torch.autograd.Function):
    """DPF: dead weights can reactivate (no gradient masking)"""
    @staticmethod
    def forward(ctx, x, mask):
        # dynamic: no need to save mask for backward since grad is unchanged
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


################################################################
# 2) DWA (Dynamic Weight Adjustment) – 3가지 실험용 마스커
################################################################

class MaskerScalingReactivate(torch.autograd.Function):
    """
    (1) Reactivate:
        g' = g*m + alpha * g * (1-m) * f
      where f = ||w| - tau|
    """
    @staticmethod
    def forward(ctx, x, mask, alpha, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha = ctx.alpha
        f = torch.abs(torch.abs(x) - threshold)  # f = ||w| - tau|
        g_new = (grad_out * mask) + (grad_out * (1 - mask) * f * alpha)
        return g_new, None, None, None


class MaskerScalingKill(torch.autograd.Function):
    """
    (2) Kill:
        g' = beta * g * m * |w| + g * (1-m)
      활성 가중치는 |w|로 스케일해 '죽이는' 방향으로, 비활성은 평범한 grad (재활성화 효과 없음)
    """
    @staticmethod
    def forward(ctx, x, mask, beta):
        ctx.save_for_backward(mask, x)
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x = ctx.saved_tensors
        beta = ctx.beta
        b = torch.abs(x)  # |w|
        g_new = (grad_out * mask * b * beta) + (grad_out * (1 - mask))
        return g_new, None, None


class MaskerScalingKillAndReactivate(torch.autograd.Function):
    """
    (3) Kill & Reactivate (양쪽 모두):
        g' = beta * g * m * |w| + alpha * g * (1-m) * ||w| - tau|
    """
    @staticmethod
    def forward(ctx, x, mask, alpha, beta, threshold):
        ctx.save_for_backward(mask, x, threshold)
        ctx.alpha = alpha
        ctx.beta = beta
        return x * mask

    @staticmethod
    def backward(ctx, grad_out):
        mask, x, threshold = ctx.saved_tensors
        alpha, beta = ctx.alpha, ctx.beta
        abs_w = torch.abs(x)
        diff = torch.abs(abs_w - threshold)  # ||w| - tau|

        base_alive = grad_out * mask * abs_w * beta
        base_dead = grad_out * (1 - mask) * diff * alpha
        g_new = base_alive + base_dead
        return g_new, None, None, None, None


################################################################
# 3) 통합 MaskConv2d (DWA only)
#    forward_type ∈ {'reactivate', 'kill', 'kill_and_reactivate'}
#    기본값은 'kill_and_reactivate'이며, 별도의 legacy 분기는 없습니다.
################################################################

class MaskConv2d(nn.Conv2d):
    """
    통합 MaskConv2d (DWA 전용)
    forward_type을 이용해 세 가지 마스커 중 하나를 선택한다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

        # 공통: pruning mask (학습 X)
        self.mask = Parameter(torch.ones_like(self.weight), requires_grad=False)

        # DWA 설정
        self.forward_type = 'kill_and_reactivate'
        self.alpha = 1.0
        self.beta = 1.0
        self.threshold = Parameter(torch.tensor(0.05, dtype=self.weight.dtype, device=self.weight.device),
                                   requires_grad=False)

    # 편의 메서드: DWA threshold 업데이트(가중치 절대값의 p-분위수)
    def update_threshold(self, percentile: int = 50):
        with torch.no_grad():
            weight_abs = torch.abs(self.weight)
            self.threshold.data = torch.quantile(weight_abs, percentile / 100.0)

    def forward(self, x):
        ft = (self.forward_type or 'kill_and_reactivate').lower()
        if ft == "reactivate":
            masked_weight = MaskerScalingReactivate.apply(
                self.weight, self.mask, self.alpha, self.threshold
            )
        elif ft == "kill":
            masked_weight = MaskerScalingKill.apply(
                self.weight, self.mask, self.beta
            )
        elif ft == "kill_and_reactivate":
            masked_weight = MaskerScalingKillAndReactivate.apply(
                self.weight, self.mask, self.alpha, self.beta, self.threshold
            )
        else:
            raise NotImplementedError(f"Unknown forward_type: {self.forward_type}")

        # 표준 Conv
        return F.conv2d(x, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# 요약: Kill & Reactivate 마스커의 최종 수정 그래디언트는
#   G' = β · G·m·|W| + α · G·(1-m)·||W|-τ|
