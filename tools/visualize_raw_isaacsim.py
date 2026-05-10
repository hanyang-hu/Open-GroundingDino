import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_name(path: Path, kind: str):
    pattern = rf"^(?P<prefix>.+)_{kind}_(?P<frame>\d+)\.(png|npy)$"
    m = re.match(pattern, path.name)
    if not m:
        return None
    return m.group("prefix"), int(m.group("frame"))


def build_pairs(dataset_root: Path):
    rgb_dir = dataset_root / "rgb"
    bbox_dir = dataset_root / "bounding_box_2d_tight"

    rgb_map = {}
    for p in rgb_dir.glob("*.png"):
        k = parse_name(p, "rgb")
        if k is not None:
            rgb_map[k] = p

    bbox_map = {}
    for p in bbox_dir.glob("*.npy"):
        k = parse_name(p, "bounding_box_2d_tight")
        if k is not None:
            bbox_map[k] = p

    common = sorted(set(rgb_map.keys()) & set(bbox_map.keys()))
    return [(rgb_map[k], bbox_map[k]) for k in common]


def load_id_map(path: Path):
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): str(v) for k, v in data.items()}


def load_id_map_from_labels_json(dataset_root: Path):
    """
    Read IsaacSim label sidecar files:
      bounding_box_2d_tight/*_labels_*.json
    and merge semanticId -> name mapping.
    """
    labels_dir = dataset_root / "bounding_box_2d_tight"
    label_files = sorted(labels_dir.glob("*_labels_*.json"))
    merged = {}
    for p in label_files:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    merged[int(k)] = str(v)
                except Exception:
                    continue
    return merged


def draw_single(img_path: Path, npy_path: Path, id_map: dict, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = img.size

    records = np.load(npy_path, allow_pickle=False)
    for rec in records:
        sid = int(rec["semanticId"])
        x0 = float(rec["x_min"])
        y0 = float(rec["y_min"])
        x1 = float(rec["x_max"])
        y1 = float(rec["y_max"])

        x0 = max(0, min(int(round(x0)), w - 1))
        y0 = max(0, min(int(round(y0)), h - 1))
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        if x1 <= x0 or y1 <= y0:
            continue

        label = id_map.get(sid, f"id:{sid}")
        color = (0, 255, 0)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((x0, y0), label, font=font)
        else:
            tw, th = draw.textsize(label, font=font)
            tb = (x0, y0, x0 + tw, y0 + th)
        draw.rectangle(tb, fill=color)
        draw.text((x0, y0), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser("Visualize raw IsaacSim boxes from rgb + npy.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/surgical_instrument"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/vis_raw"))
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--id-map",
        type=Path,
        default=None,
        help='Optional JSON mapping semanticId to name, e.g. {"0":"needle holder"}',
    )
    parser.add_argument(
        "--use-label-json",
        action="store_true",
        help="Auto-load semanticId->name from bounding_box_2d_tight/*_labels_*.json",
    )
    args = parser.parse_args()

    pairs = build_pairs(args.dataset_root)
    if not pairs:
        raise RuntimeError("No matched rgb/npy pairs found.")

    id_map = {}
    if args.use_label_json:
        id_map.update(load_id_map_from_labels_json(args.dataset_root))
    if args.id_map is not None:
        id_map.update(load_id_map(args.id_map))
    rng = random.Random(args.seed)
    n = min(args.num_samples, len(pairs))
    sampled = rng.sample(pairs, n)

    for idx, (img_path, npy_path) in enumerate(sampled, start=1):
        out_name = f"{idx:03d}_{img_path.name}"
        out_path = args.out_dir / out_name
        draw_single(img_path, npy_path, id_map, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
