import torch.nn as nn
import torch
import numpy as np

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
# cm
class Memory_module(nn.Module):
    def __init__(self, dataset, device):
        super(Memory_module, self).__init__()
        self.dataset = dataset
        self.device = device
        self.logger_info = None

        self.video_name_all = np.array(list(self.dataset.video_info_dict.keys()))
        self.video_num = len(self.video_name_all)
        example_feature = self.dataset.video_info_dict[self.video_name_all[0]]['feature']
        if len(example_feature.shape) == 3:
            self.process_len = example_feature.shape[0]
            self.dim = example_feature.shape[-1]
        else:
            self.process_len = example_feature.shape[0]
            self.dim = example_feature.shape[-1]

        self.pseudo_labels = {
            video_name: torch.from_numpy(self.dataset.video_info_dict[video_name]['pseudo_label']).to(torch.float32)
            for video_name in self.video_name_all
        }

        self.video_names_nor_hc = np.array([
            video_name for video_name in self.video_name_all
            if self.dataset.video_info_dict[video_name]['high_confidence_norvideo'] == 1
        ])

        self.normal_memory = self.build_memory()
        self.logger_info = 'Memory module built with {} high-confidence normal videos.'.format(len(self.video_names_nor_hc))

    def _load_feature_tensor(self, video_name):
        feature = self.dataset.video_info_dict[video_name]['feature']
        if feature.dtype != np.float32:
            feature = feature.astype(np.float32, copy=False)
        feature_tensor = torch.from_numpy(feature)
        if feature_tensor.dim() == 3:
            feature_tensor = feature_tensor.mean(dim=1)
        return feature_tensor.to(self.device)

    def build_memory(self):
        memory = []
        for video_name in self.video_names_nor_hc:
            feature_tensor = self._load_feature_tensor(video_name)
            memory.append(torch.mean(feature_tensor, dim=0))

        if len(memory) == 0:
            return torch.zeros(0, self.dim, device=self.device)

        return torch.stack(memory, dim=0)

    def update_dataloader(self):
        if self.normal_memory.numel() == 0:
            return False

        dataset_name = self.dataset.dataset_name
        for video_name in self.video_name_all:
            feature_tensor = self._load_feature_tensor(video_name)
            feature_tensor = feature_tensor.unsqueeze(0)  # 1 x T x D

            if dataset_name == 'ucf-crime':
                memory_expanded = self.normal_memory.unsqueeze(1).repeat(1, feature_tensor.shape[1], 1)
                feature_expanded = feature_tensor.repeat(self.normal_memory.shape[0], 1, 1)
                dist = nn.CosineSimilarity(dim=-1, eps=1e-6)(feature_expanded, memory_expanded)
                dist = 1 - dist
            else:
                memory_expanded = self.normal_memory.unsqueeze(1)
                feature_expanded = feature_tensor
                diff = feature_expanded - memory_expanded
                dist = torch.pow((torch.sum(diff * diff, dim=-1) / self.dim), 0.5)

            dist, _ = torch.min(dist, dim=0)

            pseudo_label_tensor = self.pseudo_labels[video_name].to(self.device)
            if pseudo_label_tensor.shape[0] != dist.shape[0]:
                pseudo_label_tensor = pseudo_label_tensor[:dist.shape[0]]

            reweight = torch.exp(-torch.abs(dist - pseudo_label_tensor))
            self.dataset.video_info_dict[video_name]['reweight'] = reweight.cpu().numpy()

        return True
