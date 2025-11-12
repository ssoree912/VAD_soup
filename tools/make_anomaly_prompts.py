#!/usr/bin/env python3
"""
Generate SAM2 prompt JSON files by combining YOLO detections with Attribute-VAD scores.

Steps:
1. Load per-frame features (velocity, pose, deep) extracted by feature_extraction.py.
2. Recreate the Attribute-VAD scoring pipeline to obtain per-box anomaly scores.
3. Threshold the scores, run TAO-style robust filtering, and save prompts + debug tuples.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - faiss is required
    raise ImportError("faiss is required for nearest-neighbour scoring") from exc

# Import shared robust filter utilities
from post.robust_filter import build_tracks, _count_support, _iou_xyxy, _greedy_match


def _load_object_list(path: Path) -> List[np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return [np.asarray(x) for x in arr]


def _concat_non_empty(arrays: Sequence[np.ndarray]) -> np.ndarray:
    pieces = [np.asarray(a) for a in arrays if len(a)]
    if not pieces:
        raise ValueError(f"No non-empty arrays to concatenate from {len(arrays)} items")
    return np.concatenate(pieces, axis=0)


def _build_l2_index(vectors: np.ndarray):
    index = faiss.IndexFlatL2(vectors.shape[1])
    res = None
    if hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            res = None
    index.add(vectors.astype(np.float32))
    return index, res


def _nearest_neighbor_scores(index, vectors: np.ndarray) -> np.ndarray:
    if index is None or vectors.size == 0:
        return np.zeros((0,), dtype=np.float32)
    total = vectors.shape[0]
    k = 2 if total > 1 else 1
    distances, _ = index.search(vectors.astype(np.float32), k)
    if k == 1:
        return distances[:, 0]
    return distances[:, 1]


def _normalize(scores: np.ndarray, min_v: float, max_v: float) -> np.ndarray:
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32)
    denom = max(max_v - min_v, 1e-6)
    return np.clip((scores - min_v) / denom, 0.0, 1.0).astype(np.float32)


def _ensure_2d(array: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _align_pose_scores(
    pose_scores: np.ndarray, classes: np.ndarray, num_boxes: int, person_class: int
) -> np.ndarray:
    aligned = np.zeros((num_boxes,), dtype=np.float32)
    if pose_scores.size == 0 or num_boxes == 0:
        return aligned
    if pose_scores.shape[0] == num_boxes:
        aligned[:] = pose_scores
        return aligned
    person_mask = classes == person_class
    person_count = int(np.count_nonzero(person_mask))
    if pose_scores.shape[0] == person_count:
        aligned[person_mask] = pose_scores
        return aligned
    # Fallback: fill sequentially and warn once via caller.
    limit = min(pose_scores.shape[0], num_boxes)
    aligned[:limit] = pose_scores[:limit]
    return aligned


def _gaussian_mixture(train_velocity: np.ndarray, components: int) -> GaussianMixture:
    gm = GaussianMixture(n_components=components, random_state=0)
    gm.fit(train_velocity.astype(np.float32))
    return gm


def _score_frame_components(
    velocity: np.ndarray,
    pose: np.ndarray,
    deep: np.ndarray,
    classes: np.ndarray,
    models: Dict[str, object],
    use_pose: bool,
    use_deep: bool,
    person_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_boxes = velocity.shape[0]
    vel_scores = (
        np.zeros((0,), dtype=np.float32)
        if num_boxes == 0
        else -models["velocity_gmm"].score_samples(velocity.astype(np.float32))
    )
    vel_scores = _normalize(vel_scores, models["min_velocity"], models["max_velocity"])

    pose_aligned = np.zeros((num_boxes,), dtype=np.float32)
    if use_pose and pose.size:
        D_pose, _ = models["pose_index"].search(pose.astype(np.float32), 1)
        pose_norm = _normalize(D_pose[:, 0], models["min_pose"], models["max_pose"])
        pose_aligned = _align_pose_scores(pose_norm, classes, num_boxes, person_class)

    deep_scores = np.zeros((num_boxes,), dtype=np.float32)
    if use_deep and deep.size:
        D_deep, _ = models["deep_index"].search(deep.astype(np.float32), 1)
        deep_scores = _normalize(D_deep[:, 0], models["min_deep"], models["max_deep"])

    total = vel_scores
    if use_pose:
        total = total + pose_aligned
    if use_deep:
        total = total + deep_scores
    return total.astype(np.float32), vel_scores, pose_aligned, deep_scores


def _build_clip_ranges(lengths: np.ndarray) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = 0
    for length in lengths.astype(int).tolist():
        length = max(int(length), 0)
        end = start + length
        ranges.append((start, end))
        start = end
    return ranges


class SplitEntry(NamedTuple):
    name: str
    label: int
    frames: int


def _load_split_entries(split_file: Path) -> List[SplitEntry]:
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")
    entries: List[SplitEntry] = []
    with open(split_file, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 3:
                raise ValueError(f"Malformed split line: {line}")
            name = parts[0]
            label = int(parts[1])
            frames = int(parts[2])
            entries.append(SplitEntry(name=name, label=label, frames=frames))
    if not entries:
        raise ValueError(f"No entries parsed from split file: {split_file}")
    return entries


def _frame_sort_key(key) -> Tuple[int, str]:
    try:
        return (0, f"{int(key):010d}")
    except (TypeError, ValueError):
        return (1, str(key))


def _center_xyxy(box: np.ndarray) -> List[float]:
    return [float(0.5 * (box[0] + box[2])), float(0.5 * (box[1] + box[3]))]


# Note: _iou_xyxy, _greedy_match, build_tracks, and _count_support are now imported from post.robust_filter


def _robust_prompts(
    frames_boxes: List[np.ndarray],
    per_frame_tids: List[List[int]],
    anomaly_mask: List[np.ndarray],
    frame_components: List[Dict[str, np.ndarray]],
    frame_keys: List[str],
    frame_order_to_global: List[int],
    save_interval: int,
    window: int,
    iou_thr: float,
    min_hits: int,
    frame_file_lookup: Dict[str, Optional[str]],
) -> List[Dict[str, object]]:
    prompts: List[Dict[str, object]] = []
    for t, boxes in enumerate(frames_boxes):
        if save_interval > 0 and (t % save_interval) != 0:
            continue
        mask = anomaly_mask[t]
        if not np.any(mask):
            continue
        tids = per_frame_tids[t]
        comps = frame_components[t]
        key = frame_keys[t]
        frame_file = frame_file_lookup.get(key)
        for det_idx, flag in enumerate(mask):
            if not flag:
                continue
            tid = tids[det_idx]
            if tid < 0:
                continue
            support_prev = _count_support(
                frames_boxes, per_frame_tids, tid, t, det_idx, window, iou_thr, direction=-1
            )
            support_next = _count_support(
                frames_boxes, per_frame_tids, tid, t, det_idx, window, iou_thr, direction=1
            )
            if support_prev < min_hits and support_next < min_hits:
                continue
            box = boxes[det_idx]
            prompt = {
                "frame_order": t,
                "frame_index": frame_order_to_global[t],
                "frame_key": key,
                "frame_file": frame_file,
                "bbox": box.astype(float).tolist(),
                "center": _center_xyxy(box),
                "track_id": int(tid),
                "score": float(comps["total"][det_idx]),
                "components": {
                    "velocity": float(comps["velocity"][det_idx]),
                    "pose": float(comps["pose"][det_idx]),
                    "deep": float(comps["deep"][det_idx]),
                },
            }
            prompts.append(prompt)
    return prompts


def _index_frame_files(frames_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not frames_dir.exists():
        return mapping
    for path in frames_dir.iterdir():
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        mapping[path.stem] = path.name
    return mapping


def _resolve_frame_name(mapping: Dict[str, str], key: str) -> Optional[str]:
    if key in mapping:
        return mapping[key]
    no_zero = key.lstrip("0")
    if no_zero and no_zero in mapping:
        return mapping[no_zero]
    padded = key.zfill(6)
    if padded in mapping:
        return mapping[padded]
    return None


def parse_args():
    ap = argparse.ArgumentParser("Attribute-VAD based anomaly prompt builder")
    ap.add_argument("--dataset_name", type=str, default="shanghaitech")
    ap.add_argument("--split", type=str, default="testing", choices=["training", "testing"])
    ap.add_argument("--data_root", type=str, default="./data/shanghaitech")
    ap.add_argument("--extracted_root", type=str, default="./extracted_features")
    ap.add_argument("--detections_dir", type=str, required=True,
                    help="Directory containing per-video detections.npy files for the split")
    ap.add_argument("--frames_root", type=str, default=None,
                    help="Override frames directory (defaults to data_root/<split>/frames)")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--videos", nargs="+", default=None, help="Optional subset of video folders to process")
    ap.add_argument("--tau", type=float, default=None, help="Manual anomaly threshold (overrides percentile)")
    ap.add_argument("--tau_percentile", type=float, default=97.5,
                    help="Percentile from normal (train) scores to set tau when unspecified")
    ap.add_argument("--k", type=int, default=10, help="Temporal window (frames) for robustness filtering")
    ap.add_argument("--h", type=float, default=0.5, help="IoU threshold used by filtering")
    ap.add_argument("--m", type=int, default=3, help="Required matches within window")
    ap.add_argument("--l", type=int, default=5, help="Save interval for prompts")
    ap.add_argument("--track_iou", type=float, default=0.5, help="IoU threshold for greedy track linking")
    ap.add_argument("--person_class", type=int, default=0, help="Class index used for person / pose alignment")
    ap.add_argument("--disable_pose", action="store_true", help="Skip pose contribution")
    ap.add_argument("--enable_deep_features", action="store_true",
                    help="Force using CLIP features even on ShanghaiTech")
    ap.add_argument("--disable_deep_features", action="store_true", help="Skip deep features entirely")
    ap.add_argument("--train_split_file", type=str, default="./data/shanghaitech/train_split.txt")
    ap.add_argument("--test_split_file", type=str, default="./data/shanghaitech/test_split.txt")
    return ap.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.data_root)
    extracted_root = Path(args.extracted_root) / args.dataset_name
    detections_dir = Path(args.detections_dir)
    frames_root = (
        Path(args.frames_root)
        if args.frames_root
        else dataset_dir / args.split / "frames"
    )
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    split_tag = "train" if args.split == "training" else "test"
    train_tag = "train"

    split_file = Path(args.train_split_file if split_tag == "train" else args.test_split_file)
    train_split_entries = _load_split_entries(Path(args.train_split_file))
    target_split_entries = _load_split_entries(split_file)
    clip_lengths = np.array([entry.frames for entry in target_split_entries], dtype=np.int32)
    clip_ranges = _build_clip_ranges(clip_lengths)
    video_names = [entry.name for entry in target_split_entries]

    expected_train_frames = int(sum(entry.frames for entry in train_split_entries))
    expected_target_frames = int(sum(entry.frames for entry in target_split_entries))

    def _empty_feature_sequence(length: int) -> List[np.ndarray]:
        return [np.zeros((0, 0), dtype=np.float32) for _ in range(length)]

    default_use_deep = args.dataset_name != "shanghaitech"
    if args.enable_deep_features:
        use_deep = True
    elif args.disable_deep_features:
        use_deep = False
    else:
        use_deep = default_use_deep
    use_pose = not args.disable_pose

    train_vel_path = extracted_root / train_tag / "velocity.npy"
    if not train_vel_path.exists():
        raise FileNotFoundError(f"Missing training velocity features: {train_vel_path}")
    train_velocity = _load_object_list(train_vel_path)
    if len(train_velocity) != expected_train_frames:
        raise ValueError(
            f"Train velocity frame count mismatch: split={expected_train_frames}, features={len(train_velocity)}"
        )
    train_velocity_concat = _concat_non_empty(train_velocity)

    if use_pose:
        train_pose_path = extracted_root / train_tag / "pose.npy"
        if not train_pose_path.exists():
            raise FileNotFoundError(f"Missing training pose features: {train_pose_path}")
        train_pose = _load_object_list(train_pose_path)
        train_pose_concat = _concat_non_empty(train_pose)
    else:
        train_pose = _empty_feature_sequence(expected_train_frames)
        train_pose_concat = np.zeros((0, 0), dtype=np.float32)

    if use_deep:
        train_deep_path = extracted_root / train_tag / "deep_features.npy"
        if not train_deep_path.exists():
            raise FileNotFoundError(f"Missing training deep features: {train_deep_path}")
        train_deep = _load_object_list(train_deep_path)
        train_deep_concat = _concat_non_empty(train_deep)
    else:
        train_deep = _empty_feature_sequence(expected_train_frames)
        train_deep_concat = np.zeros((0, 0), dtype=np.float32)

    test_vel_path = extracted_root / split_tag / "velocity.npy"
    if not test_vel_path.exists():
        raise FileNotFoundError(f"Missing {args.split} velocity features: {test_vel_path}")
    test_velocity = _load_object_list(test_vel_path)
    if len(test_velocity) != expected_target_frames:
        raise ValueError(
            f"{args.split} velocity frame count mismatch: split={expected_target_frames}, features={len(test_velocity)}"
        )

    if use_pose:
        test_pose_path = extracted_root / split_tag / "pose.npy"
        if not test_pose_path.exists():
            raise FileNotFoundError(f"Missing {args.split} pose features: {test_pose_path}")
        test_pose = _load_object_list(test_pose_path)
    else:
        test_pose = _empty_feature_sequence(expected_target_frames)

    if use_deep:
        test_deep_path = extracted_root / split_tag / "deep_features.npy"
        if not test_deep_path.exists():
            raise FileNotFoundError(f"Missing {args.split} deep features: {test_deep_path}")
        test_deep = _load_object_list(test_deep_path)
    else:
        test_deep = _empty_feature_sequence(expected_target_frames)

    components = 2 if args.dataset_name == "ped2" else 5
    velocity_gmm = _gaussian_mixture(train_velocity_concat, components)
    train_velocity_scores = -velocity_gmm.score_samples(train_velocity_concat)
    min_velocity = float(np.min(train_velocity_scores))
    max_velocity = float(np.percentile(train_velocity_scores, 99.9))

    pose_index = None
    min_pose = 0.0
    max_pose = 1.0
    if use_pose and train_pose_concat.size:
        pose_index, _ = _build_l2_index(train_pose_concat.astype(np.float32))
        pose_self_scores = _nearest_neighbor_scores(pose_index, train_pose_concat)
        if pose_self_scores.size:
            min_pose = float(np.min(pose_self_scores))
            max_pose = float(np.percentile(pose_self_scores, 99.9))

    deep_index = None
    min_deep = 0.0
    max_deep = 1.0
    if use_deep and train_deep_concat.size:
        deep_index, _ = _build_l2_index(train_deep_concat.astype(np.float32))
        deep_self_scores = _nearest_neighbor_scores(deep_index, train_deep_concat)
        if deep_self_scores.size:
            min_deep = float(np.min(deep_self_scores))
            max_deep = float(np.percentile(deep_self_scores, 99.9))

    models = {
        "velocity_gmm": velocity_gmm,
        "min_velocity": min_velocity,
        "max_velocity": max_velocity,
        "pose_index": pose_index,
        "min_pose": min_pose,
        "max_pose": max_pose,
        "deep_index": deep_index,
        "min_deep": min_deep,
        "max_deep": max_deep,
    }

    train_classes = _load_object_list(dataset_dir / f"{args.dataset_name}_bboxes_train_classes.npy")
    target_suffix = "test" if split_tag == "test" else "train"
    test_classes = _load_object_list(dataset_dir / f"{args.dataset_name}_bboxes_{target_suffix}_classes.npy")
    if len(train_classes) != len(train_velocity):
        raise ValueError("Train classes/velocity length mismatch. Did you regenerate bboxes?")
    if len(test_classes) != len(test_velocity):
        raise ValueError("Test classes/velocity length mismatch. Ensure convert_yolo_detections.py was run.")

    train_features = {
        "velocity": train_velocity,
        "pose": train_pose,
        "deep": train_deep,
    }
    test_features = {"velocity": test_velocity, "pose": test_pose, "deep": test_deep}

    def _score_split(features: Dict[str, List[np.ndarray]], classes_list: List[np.ndarray], store_components: bool):
        outputs = []
        pose_warned = False
        for idx, vel in enumerate(tqdm(features["velocity"], desc=f"Scoring {len(features['velocity'])} frames ({'store' if store_components else 'train'})")):
            vel_arr = _ensure_2d(vel, train_velocity_concat.shape[1])
            pose_arr = _ensure_2d(features["pose"][idx], train_pose_concat.shape[1])
            deep_arr = _ensure_2d(features["deep"][idx], train_deep_concat.shape[1])
            cls_arr = np.asarray(classes_list[idx], dtype=np.int32)
            if cls_arr.shape[0] != vel_arr.shape[0]:
                if cls_arr.size == 0:
                    cls_arr = np.zeros((vel_arr.shape[0],), dtype=np.int32)
                else:
                    cls_arr = np.resize(cls_arr, vel_arr.shape[0])
                pose_warned = True
            total, vel_scores, pose_scores, deep_scores = _score_frame_components(
                vel_arr, pose_arr, deep_arr, cls_arr, models, use_pose, use_deep, args.person_class
            )
            if store_components:
                outputs.append(
                    {
                        "total": total,
                        "velocity": vel_scores,
                        "pose": pose_scores if use_pose else np.zeros_like(total),
                        "deep": deep_scores if use_deep else np.zeros_like(total),
                        "classes": cls_arr,
                    }
                )
            else:
                outputs.append(total)
        if pose_warned:
            print("[warn] Pose/class alignment mismatch encountered; applied fallback ordering.")
        return outputs

    train_scores = _score_split(train_features, train_classes, store_components=False)
    test_scores = _score_split(test_features, test_classes, store_components=True)

    if args.tau is not None:
        tau = args.tau
        tau_source = "manual"
    else:
        flat_scores = np.concatenate([s for s in train_scores if len(s)], axis=0)
        tau = float(np.percentile(flat_scores, args.tau_percentile))
        tau_source = f"train_percentile_{args.tau_percentile}"
    print(f"[info] Using tau={tau:.4f} ({tau_source}) with pose={use_pose}, deep={use_deep}")

    detection_dirs = {d.name: d for d in detections_dir.iterdir() if d.is_dir()}
    requested = set(args.videos) if args.videos else None
    summary = []
    for vid_idx, (video_name, rng) in enumerate(zip(video_names, clip_ranges)):
        if requested and video_name not in requested:
            continue
        det_dir = detection_dirs.get(video_name)
        if det_dir is None:
            print(f"[warn] Missing detections folder for {video_name}; skipping.")
            continue
        start, end = rng
        det_file = det_dir / "detections.npy"
        if not det_file.exists():
            print(f"[warn] Missing detections for {video_name}, skipping.")
            continue
        det_map = np.load(det_file, allow_pickle=True).item()
        frame_items = sorted(det_map.items(), key=lambda kv: _frame_sort_key(kv[0]))
        expected_frames = end - start
        if expected_frames != len(frame_items):
            print(f"[warn] Frame count mismatch for {video_name}: expected {expected_frames}, got {len(frame_items)}. Skipping video.")
            continue
        frames_boxes: List[np.ndarray] = []
        frame_components: List[Dict[str, np.ndarray]] = []
        frame_keys: List[str] = []
        frame_ord_to_global: List[int] = []
        anomaly_mask: List[np.ndarray] = []

        frames_dir = frames_root / video_name
        frame_lookup = _index_frame_files(frames_dir)
        processed = True
        for local_idx, (frame_key, entry) in enumerate(frame_items):
            global_idx = start + local_idx
            boxes = np.asarray(entry["boxes"], dtype=np.float32)
            test_box_arr = np.asarray(test_features["velocity"][global_idx])
            if boxes.shape[0] != test_box_arr.shape[0]:
                print(f"[warn] Detection/VAD bbox count mismatch at {video_name} frame {frame_key}; skipping video.")
                processed = False
                break
            comp = test_scores[global_idx]
            mask = comp["total"] > tau
            frames_boxes.append(boxes)
            frame_components.append(comp)
            frame_keys.append(str(frame_key))
            frame_ord_to_global.append(global_idx)
            anomaly_mask.append(mask)
        if not processed:
            continue

        per_frame_tids = build_tracks(frames_boxes, args.track_iou)
        prompts = _robust_prompts(
            frames_boxes=frames_boxes,
            per_frame_tids=per_frame_tids,
            anomaly_mask=anomaly_mask,
            frame_components=frame_components,
            frame_keys=frame_keys,
            frame_order_to_global=frame_ord_to_global,
            save_interval=args.l,
            window=args.k,
            iou_thr=args.h,
            min_hits=args.m,
            frame_file_lookup={k: _resolve_frame_name(frame_lookup, k) for k in frame_keys},
        )

        video_out = out_root / args.split / video_name
        video_out.mkdir(parents=True, exist_ok=True)
        out_json = video_out / "robust_prompts.json"
        payload = {
            "dataset": args.dataset_name,
            "split": args.split,
            "video": video_name,
            "frames_dir": str(frames_dir),
            "frame_keys": frame_keys,
            "params": {
                "tau": tau,
                "tau_source": tau_source,
                "k": args.k,
                "h": args.h,
                "m": args.m,
                "l": args.l,
                "track_iou": args.track_iou,
                "use_pose": use_pose,
                "use_deep": use_deep,
            },
            "prompts": prompts,
        }
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        tuples = [
            (
                p["frame_key"],
                np.array(p["bbox"], dtype=np.float32),
                np.array(p["center"], dtype=np.float32),
                p["track_id"],
                p["score"],
            )
            for p in prompts
        ]
        np.save(video_out / "filtered_boxes.npy", np.array(tuples, dtype=object))
        summary.append((video_name, len(prompts)))
        print(f"[save] {out_json}  (#prompts={len(prompts)})")

    if not summary:
        print("[warn] No videos were processed; check --videos or input paths.")
    else:
        print("=== Prompt summary ===")
        for name, count in summary:
            print(f"{name}: {count}")


if __name__ == "__main__":
    main()
