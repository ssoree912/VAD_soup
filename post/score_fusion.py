import numpy as np


def _align_lengths(a: np.ndarray, b: np.ndarray):
    """Align two 1D arrays by truncating to the minimum length."""
    if a.ndim != 1:
        a = a.reshape(-1)
    if b.ndim != 1:
        b = b.reshape(-1)
    if len(a) == len(b):
        return a, b
    m = min(len(a), len(b))
    return a[:m], b[:m]


def fuse_scores(
    snippet_scores: dict,
    roi_scores: dict | None,
    method: str = "weighted",
    alpha: float = 0.5,
):
    """
    Fuse LANP snippet scores with ROI-based scores.

    Args:
        snippet_scores: dict[video] -> 1D np.ndarray (LANP scores)
        roi_scores: dict[video] -> 1D np.ndarray (ROI scores)
        method: "weighted" (default), "mean", "max", "sum"
        alpha: weight for ROI scores when method="weighted"

    Returns:
        dict[video] -> fused scores (1D np.ndarray)
    """
    if roi_scores is None:
        return snippet_scores

    fused = {}
    method = (method or "weighted").lower()
    for vid, lanp_arr in snippet_scores.items():
        if vid not in roi_scores:
            fused[vid] = lanp_arr
            continue
        roi_arr = roi_scores[vid]
        lanp_arr = np.asarray(lanp_arr, dtype=np.float32).reshape(-1)
        roi_arr = np.asarray(roi_arr, dtype=np.float32).reshape(-1)
        lanp_arr, roi_arr = _align_lengths(lanp_arr, roi_arr)

        if method == "mean":
            fused_arr = 0.5 * (lanp_arr + roi_arr)
        elif method == "max":
            fused_arr = np.maximum(lanp_arr, roi_arr)
        elif method == "sum":
            fused_arr = lanp_arr + roi_arr
        else:  # "weighted" or unknown -> default weighted
            w = float(alpha)
            fused_arr = (1.0 - w) * lanp_arr + w * roi_arr

        fused[vid] = fused_arr

    return fused