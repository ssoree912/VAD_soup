## TAO ↔ LANP-UVAD Integration Guide

This document maps the experiment plan (“TAO → LANP-UVAD”) to concrete code entry
points inside this repository and lists the remaining hooks that need to be
implemented for a fully automated run.

### 0. Goals

| Flow | Implementation Anchor | Notes |
| --- | --- | --- |
| Frame-level LANP baseline | `main.py`, `model.AN_Model`, `config/*.yaml` | Stays untouched. |
| Object & pixel localization | `lanp/backbone.py`, `features/roi_pool.py`, `detector/yolo.py`, `post/robust_filter.py`, `segment/sam2_runner.py` | New modules to plug TAO pipeline. |
| Training enhancements (LANP propagation + re-weighting) | `lanp/train/pseudo_label.py`, `lanp/train/reweight.py`, `lanp/memory.py` | Provide hooks for B1/B2 ablations. |

### 1. Datasets & Splits

- Use the existing dataset loaders in `data/dataset_sh.py` and `data/dataset_ucf.py`.
- Object/pixel add-ons consume frame folders under `data/<dataset>/<split>/frames`.

### 2. Baselines

- **B0 (LANP-UVAD)**: `main.py` + `config/config_*.yaml`.
- **B1 (TAO-style inference)**:
  - Run YOLO detections via `detection/yolo_detection.py` (wraps `detector.YOLODetector`).
  - Build per-box features with `lanp/backbone.BackboneFeatureExtractor` + `features.roi_pool.roi_feature_vectors`.
  - Robust filtering + SAM2 prompts via `post.robust_filter` and `segment.SAM2Runner`.

### 3. Combined Experiment Arms

| Step | Purpose | Code Hooks / Scripts |
| --- | --- | --- |
| A1 Frame Gating | select top-P% LANP frames | `main.py --eval_only --save_frame_scores_path ...`, `tools/frame_gate.py` |
| A2 ROI scoring | compute `d_i^t` using memory bank | `detection/yolo_detection.py`, `features/compute_roi_scores.py`, `lanp/memory.MemoryModule` |
| A3 Robust filtering | TAO inheritance/consistency | `post/generate_robust_prompts.py`, `post.robust_filter.robust_filter` |
| A4 SAM2 masks | prompt-based segmentation | `segment/sam2_runner.py` CLI |
| A5 Score fusion | update frame scores with ROI max | `post/score_fusion.py`, `lanp/train/reweight.blend_scores` |
| B1 Pseudo labels | upgrade anomaly candidates | `lanp/train.pseudo_label.merge_frame_masks` + `reinforce_pseudo_labels` |
| B2 Re-weighting | mix object/global distances | `lanp/train.reweight.reduce_roi_scores` + `blend_scores` + `update_reweight` |

### 4. Thresholds & Hyper-Parameters

- Frame gating `P` and box top-`q` are arguments to `gate_frames_by_score` / `gate_frames_by_boxes`.
- Robust-filter config (`k/m/h`) is represented by `post.robust_filter.RobustFilterConfig`.
- `SAM2Runner.run(... use_box_fallback=True)` enables quick dry-runs without SAM weights.

### 5. Metrics

- `eval/metrics.py` provides:
  - Frame/snippet ROC-AUC & AP (`frame_metrics`, `snippet_metrics`).
  - Pixel AUROC/AP/AUPRO + F1 sweep (`pixel_metrics`, `pixel_f1_at_threshold`).
  - Object metrics (`compute_rbdc`, `compute_tbdc` with `eval.metrics.Track`).

### 6. Minimum Ablations

Use the functions above to toggle:

1. `A*` vs baseline by skipping the ROI pipeline.
2. Score fusion by comparing original frame scores vs `lanp/train/reweight.blend_scores`.
3. B1/B2 by enabling `reinforce_pseudo_labels` and `update_reweight`.
4. ROI scoring alternatives by swapping `roi_feature_vectors` reducer/normalizer.
5. Robust filter via `RobustFilterConfig(min_hits=0)` for the ablation.
6. SAM2 propagation interval by subsampling prompts before `SAM2Runner`.

### 7. Implementation Checklist

| Module | Responsibility |
| --- | --- |
| `lanp/backbone.py` | Shared feature extractor (frame/snippet parity). |
| `lanp/memory.py` | Normality memory (moved from `model.py`). |
| `detector/yolo.py` | Import-friendly YOLOv5 wrapper. |
| `features/roi_pool.py` | ROI Align helper for ROI feature extraction. |
| `post/robust_filter.py` | TAO-style spatial-temporal filtering. |
| `segment/sam2_runner.py` | Prompt-to-mask utility (SAM2 or fallback). |
| `lanp/train/pseudo_label.py` | Frame gating + pseudo-label reinforcement. |
| `lanp/train/reweight.py` | Object/global score fusion for loss re-weighting. |
| `eval/metrics.py` | Unified metrics (frame, object, pixel). |

### 8. Next Steps

1. `python main.py --config config/config_shanghaitech.yaml --eval_only \\
   --save_frame_scores_path results/scores.npy \\
   --save_snippet_scores_path results/snippet_scores.npy \\
   --save_memory_path results/normal_memory.npz`  → dump LANP frame/snippet scores + memory.
2. `python tools/frame_gate.py --scores results/scores.npy --percentile 95 \\
   --output_txt gated_frames.txt --output_json gated_frames.json`  → top-P% frame gating.
3. `python detection/yolo_detection.py --data_root ./data/shanghaitech --split test \\
   --gated_frames gated_frames.txt --output_root artifacts/detections`  → YOLO on gated frames only.
4. `python features/compute_roi_scores.py --detections_root artifacts/detections/shanghaitech/test \\
   --frames_root data/shanghaitech/test/frames --memory_path results/normal_memory.npz \\
   --output_features artifacts/features/shanghaitech/test/deep_features.npy \\
   --output_scores artifacts/roi_scores_sh.npz --seg_len 32`  → ROI features + cosine scores.
5. `python post/generate_robust_prompts.py --detections_root artifacts/detections/shanghaitech/test \\
   --roi_scores artifacts/roi_scores_sh.npz --frames_root data/shanghaitech/test/frames \\
   --output_root artifacts/prompts --score_threshold 0.5 --window 5 --iou_threshold 0.3 --min_hits 3`  → TAO prompts.
6. `python -m segment.sam2_runner --prompts artifacts/prompts/<video>/robust_prompts.json \\
   --out-dir masks/<video>/ --box-fallback`  → SAM2 (or box fallback) masks.
7. `python main.py --config config/config_shanghaitech.yaml --roi_scores_path artifacts/roi_scores_sh.npz \\
   --roi_score_fusion max --roi_reweight_lambda 0.5`  → score fusion + re-weighting.
