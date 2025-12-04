import numpy as np, os, json
scores = np.load('data/ucf-crime/pseudo_label_scores_ucf.npy', allow_pickle=True).item()
miss = []
for line in open('data/ucf-crime/Anomaly_Train.txt'):
    vid = line.strip().split(',')[0]
    z_len = len(scores[vid]['pseudo_label_scores'])
    feat = np.load(f'data/ucf-crime/features/{vid}_res.npy')
    miss.append((vid, z_len, len(feat)))
print([m for m in miss if m[1]!=m[2]][:20])
