import torch.nn as nn
import torch

from lanp.memory import Memory_module
class AD_Model(nn.Module):
    def __init__(self, len_feature, feature_embed, dropout_rate=0.6):
        super(AD_Model, self).__init__()
        self.len_feature = len_feature
        self.feature_embed = feature_embed
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=self.feature_embed, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Linear(self.feature_embed, 512), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(512, 1), nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        bs, ncrops, t, f = x.shape
        embeddings = x.view(-1, t, f)
        embeddings = embeddings.permute(0, 2, 1) # [b, t, d] --> [b, d, t]
        embeddings = self.f_embed(embeddings) # temporal convolution
        embeddings = embeddings.permute(0, 2, 1) # [b, d, t] --> [b, t, d]
        out = self.dropout(embeddings)
        out = self.f_cls(out)
        logits = out.view(bs, ncrops, -1).mean(1)

        return logits
