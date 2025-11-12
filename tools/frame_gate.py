#!/usr/bin/env python3
"""Select top-P% LANP frames and dump lists for downstream gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from lanp.train import gate_frames_by_score


def load_frame_scores(path: Path) -> Dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    data: Dict[str, np.ndarray]
    if isinstance(payload, np.lib.npyio.NpzFile):
        candidate = None
        for key in payload.files:
            arr = payload[key]
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                candidate = arr.flat[0]
                break
        if isinstance(candidate, dict):
            data = candidate
        else:
            data = {k: payload[k] for k in payload.files}
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        data = payload.item()
    elif isinstance(payload, dict):
        data = payload
    else:
        raise ValueError(f"Unsupported score container: {type(payload)}")
    scores = {k: np.asarray(v).reshape(-1) for k, v in data.items()}
    return scores


def save_txt(path: Path, gated: Dict[str, np.ndarray]):
    lines: List[str] = []
    for video, mask in sorted(gated.items()):
        idxs = np.flatnonzero(mask)
        for idx in idxs:
            lines.append(f"{video} {int(idx)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_json(path: Path, gated: Dict[str, np.ndarray]):
    payload = {video: np.flatnonzero(mask).astype(int).tolist() for video, mask in gated.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Gate frames by percentile of LANP scores")
    parser.add_argument("--scores", required=True, help="Path to frame-level score npy/npz file")
    parser.add_argument("--percentile", type=float, default=95.0,
                        help="Keep frames above this percentile (default: 95, i.e., top 5%)")
    parser.add_argument("--output_txt", type=str, default=None, help="Optional txt file with 'video frame' per line")
    parser.add_argument("--output_json", type=str, default=None, help="Optional JSON dump of gated frames")
    return parser.parse_args()


def main():
    args = parse_args()
    score_map = load_frame_scores(Path(args.scores))
    gated = gate_frames_by_score(score_map, top_percent=args.percentile)
    if not args.output_txt and not args.output_json:
        raise ValueError("At least one output (txt/json) must be specified.")
    if args.output_txt:
        save_txt(Path(args.output_txt), gated)
    if args.output_json:
        save_json(Path(args.output_json), gated)
    print(f"[gate] processed {len(gated)} videos")


if __name__ == "__main__":
    main()
