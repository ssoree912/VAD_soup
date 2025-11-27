#!/usr/bin/env python3
"""LANP 게이팅 + Attn U-Net 에러맵 + (옵션) YOLO-World 객체 필터링."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.att_unet import AttUNetPredictor  # noqa: E402
from models.yolo_world import DEFAULT_PROMPTS, YOLOWorldDetector, YOLOWorldOutputs, load_prompts  # noqa: E402
from tools.ae_heatmap_to_detections import mask_to_boxes_and_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LANP top-p% 프레임에 대해 Attn U-Net 에러맵을 계산하고, "
        "옵션으로 YOLO-World와 결합하여 객체 단위 이상 박스를 생성."
    )
    p.add_argument("--frames_root", required=True, help="비디오별 프레임 폴더 루트.")
    p.add_argument("--lanp_scores", required=True, help="LANP frame score npz/npy 경로.")
    p.add_argument("--predictor_ckpt", required=True, help="학습된 Attn U-Net 체크포인트.")
    p.add_argument("--output_root", required=True, help="에러맵을 저장할 루트 (<video>/<frame>_err.npy).")
    p.add_argument(
        "--detections_root",
        default=None,
        help="detection.npy를 저장할 루트. 지정하지 않으면 에러맵만 저장.",
    )

    p.add_argument("--t", type=int, default=4, help="문맥 프레임 개수.")
    p.add_argument("--image_size", type=int, default=256, help="모델 입력 해상도.")
    p.add_argument("--top_percent", type=float, default=20.0, help="LANP 상위 p%% 프레임만 처리.")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda 등 디바이스 지정.")
    p.add_argument("--base_channels", type=int, default=64, help="Attn U-Net 베이스 채널.")

    # 에러맵 → 픽셀/블랍 박스
    p.add_argument("--err_percentile", type=float, default=97.0, help="에러맵 퍼센타일 임계값.")
    p.add_argument("--min_area", type=int, default=30, help="마스크 최소 영역(px).")
    p.add_argument("--morph_kernel", type=int, default=0, help="마스크 morphology 커널(0이면 비활성).")
    p.add_argument("--score_agg", choices=["mean", "max"], default="mean", help="에러맵 박스 스코어 산출 방식.")
    p.add_argument("--class_id", type=int, default=0, help="YOLO 미사용 시 고정 클래스 id.")

    # YOLO-World 옵션
    p.add_argument("--use_yolo", action="store_true", help="YOLO-World를 사용해 객체 기반 필터링.")
    p.add_argument("--yolo_model", type=str, default="yoloworld_l", help="YOLO-World 가중치 이름/경로.")
    p.add_argument("--yolo_device", type=str, default=None, help="YOLO-World 전용 디바이스.")
    p.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO-World confidence threshold.")
    p.add_argument("--yolo_iou", type=float, default=0.45, help="YOLO-World NMS IoU threshold.")
    p.add_argument("--yolo_max_det", type=int, default=200, help="YOLO-World 최대 detection 수.")
    p.add_argument("--yolo_imgsz", type=int, default=640, help="YOLO-World 입력 해상도.")
    p.add_argument("--yolo_prompts", type=str, default=None, help="콤마로 구분한 프롬프트 문자열.")
    p.add_argument("--yolo_prompt_file", type=str, default=None, help="프롬프트 txt 파일 경로(줄당 1개).")

    # YOLO + 에러맵 결합 기준
    p.add_argument("--ratio_thr", type=float, default=0.15, help="박스 내부 high-error 비율 최소값.")
    p.add_argument("--score_thr", type=float, default=0.0, help="박스 내부 에러 스코어 최소값.")
    p.add_argument("--lanp_score_thr", type=float, default=None, help="LANP 점수 하한(없으면 비활성).")
    p.add_argument("--obj_box_mode", choices=["full", "refined"], default="full",
                   help="full=YOLO 박스 유지, refined=YOLO∩에러마스크 블랍으로 재계산.")
    return p.parse_args()


def load_frame_scores(path: Path) -> Dict[str, np.ndarray]:
    """LANP 점수 npy/npz 로드 (dict 형태 지원)."""
    payload = np.load(path, allow_pickle=True)
    data = None

    if isinstance(payload, np.lib.npyio.NpzFile):
        if "data" in payload.files:
            arr = payload["data"]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                maybe_dict = arr.flat[0]
                if isinstance(maybe_dict, dict):
                    data = maybe_dict
        else:
            data = {k: np.asarray(payload[k]) for k in payload.files}
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        obj = payload.item()
        if isinstance(obj, dict):
            data = obj

    if data is None:
        raise ValueError(f"Unsupported LANP frame score format in {path}")
    return data


def load_predictor(ckpt_path: Path, device: torch.device, t: int, base_ch: int) -> AttUNetPredictor:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = AttUNetPredictor(t=t, base_ch=base_ch).to(device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def make_err_mask(err: np.ndarray, thr: float, morph_kernel: int, min_area: int) -> np.ndarray:
    """에러맵을 threshold + morphology + 영역 필터링으로 바이너리 마스크로 변환."""
    mask = (err >= thr).astype(np.uint8)

    if morph_kernel > 0:
        k = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 1 and mask.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        cleaned = np.zeros_like(mask)
        for idx in keep:
            cleaned[labels == idx] = 1
        mask = cleaned
    return mask


def rescale_boxes(boxes: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] *= scale_x
    out[:, [1, 3]] *= scale_y
    return out


def unet_only_boxes(
    err: np.ndarray,
    thr_err: float,
    min_area: int,
    morph_kernel: int,
    score_agg: str,
    class_id: int,
    orig_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, det_scores = mask_to_boxes_and_scores(
        err,
        thr=thr_err,
        min_area=min_area,
        score_agg=score_agg,
        morph_kernel=morph_kernel,
    )
    if boxes.shape[0] == 0:
        return boxes, det_scores, np.zeros((0,), dtype=np.int64)

    orig_w, orig_h = orig_size
    sx = orig_w / float(err.shape[1])
    sy = orig_h / float(err.shape[0])
    boxes_scaled = rescale_boxes(boxes, sx, sy)
    classes = np.full((boxes.shape[0],), class_id, dtype=np.int64)
    return boxes_scaled, det_scores, classes


def _mask_to_refined_boxes(
    mask_roi: np.ndarray,
    offset_xy: Tuple[int, int],
    inv_scale: Tuple[float, float],
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """YOLO 박스 내부 마스크에서 contour bbox 추출 후 원본 해상도로 복원."""
    if mask_roi.dtype != np.uint8:
        mask_roi = mask_roi.astype(np.uint8)
    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 4), dtype=np.float32)

    ox, oy = offset_xy
    inv_sx, inv_sy = inv_scale
    orig_w, orig_h = orig_size

    boxes: List[List[float]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x1 = (ox + x) * inv_sx
        y1 = (oy + y) * inv_sy
        x2 = (ox + x + w) * inv_sx
        y2 = (oy + y + h) * inv_sy
        boxes.append(
            [
                float(np.clip(x1, 0, orig_w)),
                float(np.clip(y1, 0, orig_h)),
                float(np.clip(x2, 0, orig_w)),
                float(np.clip(y2, 0, orig_h)),
            ]
        )
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def filter_yolo_with_error(
    yolo_out: YOLOWorldOutputs,
    err: np.ndarray,
    mask: np.ndarray,
    score_agg: str,
    ratio_thr: float,
    score_thr: float,
    obj_box_mode: str,
    orig_size: Tuple[int, int],
    debug: bool = False,   # 🔹 디버그 플래그 추가
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YOLO 박스를 에러맵 마스크와 겹침 비율/스코어 기준으로 필터링."""
    if yolo_out.boxes.shape[0] == 0:
        if debug:
            print("[debug][yolo_filter] no YOLO boxes.")
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    if mask.sum() == 0:
        if debug:
            print("[debug][yolo_filter] error mask is empty (mask.sum() == 0).")
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    err_h, err_w = err.shape
    orig_w, orig_h = orig_size
    sx = err_w / float(orig_w)
    sy = err_h / float(orig_h)
    inv_sx = 1.0 / sx if sx != 0 else 0.0
    inv_sy = 1.0 / sy if sy != 0 else 0.0

    keep_boxes: List[List[float]] = []
    keep_scores: List[float] = []
    keep_classes: List[int] = []

    num_boxes = yolo_out.boxes.shape[0]

    for i, (box, cls_id) in enumerate(zip(yolo_out.boxes, yolo_out.class_ids)):
        # err 해상도로 스케일
        x1e = int(np.clip(round(box[0] * sx), 0, err_w - 1))
        y1e = int(np.clip(round(box[1] * sy), 0, err_h - 1))
        x2e = int(np.clip(round(box[2] * sx), 0, err_w))
        y2e = int(np.clip(round(box[3] * sy), 0, err_h))
        if x2e <= x1e or y2e <= y1e:
            if debug:
                print(f"[debug][yolo_filter] box #{i} cls={cls_id}: invalid scaled box, skip.")
            continue

        mask_roi = mask[y1e:y2e, x1e:x2e]
        if mask_roi.size == 0 or mask_roi.sum() == 0:
            if debug:
                print(f"[debug][yolo_filter] box #{i} cls={cls_id}: no overlap with error mask, skip.")
            continue

        ratio = float(mask_roi.mean())

        err_roi = err[y1e:y2e, x1e:x2e]
        err_score = float(err_roi.max() if score_agg == "max" else err_roi.mean())

        if debug:
            print(
                f"[debug][yolo_filter] box #{i} cls={cls_id}: "
                f"ratio={ratio:.4f} (thr={ratio_thr}), "
                f"err_score={err_score:.4f} (thr={score_thr})"
            )

        if ratio < ratio_thr:
            if debug:
                print(f"  -> SKIP: ratio < ratio_thr")
            continue

        if err_score < score_thr:
            if debug:
                print(f"  -> SKIP: err_score < score_thr")
            continue

        if obj_box_mode == "refined":
            refined = _mask_to_refined_boxes(mask_roi, (x1e, y1e), (inv_sx, inv_sy), orig_size)
            if refined.shape[0] == 0:
                if debug:
                    print(f"  -> SKIP: refined boxes empty")
                continue
            keep_boxes.extend(refined.tolist())
            keep_scores.extend([err_score] * refined.shape[0])
            keep_classes.extend([int(cls_id)] * refined.shape[0])
            if debug:
                print(f"  -> KEEP: refined {refined.shape[0]} boxes")
        else:
            # full 박스는 원본 좌표를 그대로 사용 (clip 포함)
            x1o = float(np.clip(box[0], 0, orig_w))
            y1o = float(np.clip(box[1], 0, orig_h))
            x2o = float(np.clip(box[2], 0, orig_w))
            y2o = float(np.clip(box[3], 0, orig_h))
            keep_boxes.append([x1o, y1o, x2o, y2o])
            keep_scores.append(err_score)
            keep_classes.append(int(cls_id))
            if debug:
                print(f"  -> KEEP: full box")

    if debug:
        print(f"[debug][yolo_filter] kept {len(keep_boxes)} / {num_boxes} boxes.\n")

    if not keep_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.asarray(keep_boxes, dtype=np.float32),
        np.asarray(keep_scores, dtype=np.float32),
        np.asarray(keep_classes, dtype=np.int64),
    )

def ensure_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.top_percent <= 100.0):
        raise ValueError("--top_percent must be in (0, 100]")
    if args.use_yolo and args.detections_root is None:
        raise ValueError("--detections_root를 지정해야 --use_yolo를 사용할 수 있습니다.")

    device = ensure_device(args.device)
    frames_root = Path(args.frames_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    det_root = Path(args.detections_root) if args.detections_root else None
    if det_root:
        det_root.mkdir(parents=True, exist_ok=True)

    lanp_scores = load_frame_scores(Path(args.lanp_scores))
    model = load_predictor(Path(args.predictor_ckpt), device, t=args.t, base_ch=args.base_channels)

    transform = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def denorm(x: torch.Tensor) -> torch.Tensor:
        return (x * 0.5 + 0.5).clamp(0, 1)

    yolo_detector: YOLOWorldDetector | None = None
    yolo_prompts: List[str] = list(DEFAULT_PROMPTS)
    if args.use_yolo:
        yolo_prompts = load_prompts(args.yolo_prompts, args.yolo_prompt_file)
        yolo_detector = YOLOWorldDetector(
            model_name=args.yolo_model,
            prompts=yolo_prompts,
            device=args.yolo_device,
            conf=args.yolo_conf,
            iou=args.yolo_iou,
            max_det=args.yolo_max_det,
            imgsz=args.yolo_imgsz,
        )
        if det_root:
            # 추후 class id 매핑 확인용
            prompt_path = det_root / "yolo_prompts.txt"
            prompt_path.write_text("\n".join(yolo_prompts), encoding="utf-8")

    video_dirs = sorted([p for p in frames_root.iterdir() if p.is_dir()])
    for video_dir in tqdm(video_dirs, desc="Videos"):
        vid = video_dir.name
        if vid not in lanp_scores:
            print(f"[warn] missing LANP scores for video {vid}, skipping")
            continue
        scores = np.asarray(lanp_scores[vid], dtype=np.float32).reshape(-1)
        frame_paths = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
        if not frame_paths:
            print(f"[warn] no frames found for {vid}")
            continue
        frame_count = min(len(frame_paths), len(scores))
        if frame_count <= args.t:
            print(f"[warn] not enough frames for {vid} (need > {args.t})")
            continue

        thr = np.percentile(scores[:frame_count], 100.0 - args.top_percent)
        top_indices = np.nonzero(scores[:frame_count] >= thr)[0]
        print(f"[{vid}] LANP threshold top {args.top_percent:.1f}% = {thr:.6f} | candidates={len(top_indices)}")
        if top_indices.size == 0:
            continue

        video_out = out_root / vid
        video_out.mkdir(parents=True, exist_ok=True)

        boxes_seq: List[np.ndarray] = []
        scores_seq: List[np.ndarray] = []
        classes_seq: List[np.ndarray] = []
        if det_root:
            boxes_seq = [np.zeros((0, 4), dtype=np.float32) for _ in range(frame_count)]
            scores_seq = [np.zeros((0,), dtype=np.float32) for _ in range(frame_count)]
            classes_seq = [np.zeros((0,), dtype=np.int64) for _ in range(frame_count)]

        processed = 0
        skipped_prefix = 0
        for idx in tqdm(top_indices, desc=f"{vid} frames", leave=False):
            if idx < args.t:
                skipped_prefix += 1
                continue
            start = idx - args.t
            ctx_paths = frame_paths[start:idx]
            if len(ctx_paths) < args.t or idx >= frame_count:
                continue

            target_path = frame_paths[idx]
            target_img = Image.open(target_path).convert("RGB")
            orig_w, orig_h = target_img.size

            ctx_imgs = [transform(Image.open(p).convert("RGB")) for p in ctx_paths]
            target_tensor = transform(target_img)

            inp = torch.cat(ctx_imgs, dim=0).unsqueeze(0).to(device, non_blocking=True)  # (1,3t,H,W)
            tgt = target_tensor.unsqueeze(0).to(device, non_blocking=True)  # (1,3,H,W)

            with torch.no_grad():
                pred = model(inp)

            pred_denorm = denorm(pred)
            tgt_denorm = denorm(tgt)
            err = ((pred_denorm - tgt_denorm) ** 2).mean(dim=1).squeeze(0).cpu().numpy()

            np.save(video_out / f"{target_path.stem}_err.npy", err)
            processed += 1

            if det_root is None:
                continue

            thr_err = float(np.percentile(err, args.err_percentile))
            mask = make_err_mask(err, thr_err, args.morph_kernel, args.min_area)

            frame_score = float(scores[idx])
            if args.lanp_score_thr is not None and frame_score < args.lanp_score_thr:
                continue

            if args.use_yolo and yolo_detector is not None:
                yolo_out = yolo_detector.predict(np.array(target_img)[:, :, ::-1])  # BGR 포맷 허용
                boxes, det_scores, cls_ids = filter_yolo_with_error(
                    yolo_out,
                    err,
                    mask,
                    score_agg=args.score_agg,
                    ratio_thr=args.ratio_thr,
                    score_thr=args.score_thr,
                    obj_box_mode=args.obj_box_mode,
                    orig_size=(orig_w, orig_h),
                     debug=False,
                )
            else:
                boxes, det_scores, cls_ids = unet_only_boxes(
                    err,
                    thr_err,
                    args.min_area,
                    args.morph_kernel,
                    args.score_agg,
                    args.class_id,
                    orig_size=(orig_w, orig_h),
                )

            if boxes.shape[0] > 0:
                boxes_seq[idx] = boxes
                scores_seq[idx] = det_scores
                classes_seq[idx] = cls_ids

        print(f"[{vid}] processed={processed}, skipped_prefix(<t)={skipped_prefix}")

        if det_root is not None and processed > 0:
            out_video_dir = det_root / vid
            out_video_dir.mkdir(parents=True, exist_ok=True)
            frame_files = np.array([p.name for p in frame_paths[:frame_count]], dtype=object)
            frame_indices = np.arange(frame_count, dtype=np.int32)
            payload: Dict[str, np.ndarray] = {
                "boxes": np.array(boxes_seq, dtype=object),
                "scores": np.array(scores_seq, dtype=object),
                "classes": np.array(classes_seq, dtype=object),
                "frame_files": frame_files,
                "frame_indices": frame_indices,
                "num_frames": np.array(frame_count, dtype=np.int32),
            }
            if args.use_yolo:
                # YOLO / YOLO-World 모드: 프롬프트 리스트를 클래스 이름으로 저장
                payload["class_names"] = np.array(yolo_prompts, dtype=object)
            else:
                payload["class_names"] = np.array(["anomaly"], dtype=object)

            np.save(out_video_dir / "detections.npy", payload, allow_pickle=True)
            print(f"[save] detections -> {out_video_dir / 'detections.npy'}")


if __name__ == "__main__":
    main()
