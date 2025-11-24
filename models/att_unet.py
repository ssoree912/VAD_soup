import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3x3 convs with BN+ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Oktay Attention U-Net gate."""

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: encoder feat, g: decoder feat
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # upsample gate if needed
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B,1,H,W)
        return x * psi


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.att = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=out_ch)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip_att = self.att(skip, x)
        x = torch.cat([x, skip_att], dim=1)
        return self.conv(x)


class AttUNetPredictor(nn.Module):
    """
    Attention U-Net that predicts I_{t+1} from concatenated t previous frames.
    Input shape: (B, 3*t, H, W)
    Output shape: (B, 3, H, W) in [-1, 1] (tanh).
    """

    def __init__(self, t: int = 4, base_ch: int = 64) -> None:
        super().__init__()
        in_ch = 3 * t

        # encoder
        self.c1 = ConvBlock(in_ch, base_ch)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(base_ch, base_ch * 2)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(base_ch * 8, base_ch * 16)

        # decoder
        self.u4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.u3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.u2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.u1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        p4 = self.p4(c4)
        c5 = self.c5(p4)

        # decoder + attention skips
        x = self.u4(c5, c4)
        x = self.u3(x, c3)
        x = self.u2(x, c2)
        x = self.u1(x, c1)
        out = self.out_conv(x)
        return torch.tanh(out)
