import numpy as np

payload = np.load("results/normal_memory.npz", allow_pickle=True)
arr = payload["data"].item()["normal_memory"]
np.savez_compressed("results/normal_memory_flat.npz", normal_memory=arr)
