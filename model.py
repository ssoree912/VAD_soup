import torch.nn as nn
import torch

from lanp.memory import Memory_module
class AD_Model(nn.Module):
    def __init__(self, len_feature, feature_embed, dropout_rate=0.3):
        super(AD_Model, self).__init__()
        self.len_feature = len_feature
        self.feature_embed = feature_embed

        # 🔹 2-layer temporal conv + ReLU (조금 더 깊게)
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature,
                      out_channels=self.feature_embed,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.feature_embed,
                      out_channels=self.feature_embed,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # 🔹 FC 쪽도 살짝 가볍게/안정적으로
        self.f_cls = nn.Sequential(
            nn.Linear(self.feature_embed, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bs, ncrops, t, f = x.shape          # [B, Nc, T, F]
        x = x.view(-1, t, f)                # [B*Nc, T, F]
        x = x.permute(0, 2, 1)              # [B*Nc, F, T]
        x = self.f_embed(x)                 # [B*Nc, C, T]
        x = x.permute(0, 2, 1)              # [B*Nc, T, C]

        out = self.f_cls(x)                 # [B*Nc, T, 1]
        logits = out.view(bs, ncrops, t, 1).mean(1)  # [B, T, 1] crop 평균
        logits = logits.squeeze(-1)         # [B, T]

        return logits