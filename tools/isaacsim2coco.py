import argparse
import shutil
import json
import random
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize_label_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def load_label_map(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_name = {int(k): str(v) for k, v in data.items()}
    # COCO category id is 1-based
    return {i + 1: id_to_name[i] for i in sorted(id_to_name.keys())}


def load_label_conversion(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # normalized raw name -> normalized canonical name
    return {normalize_label_name(k): normalize_label_name(v) for k, v in data.items()}


def parse_name(path: Path, kind: str):
    pattern = rf"^(?P<prefix>.+)_{kind}_(?P<frame>\d+)\.(png|npy)$"
    match = re.match(pattern, path.name)
    if not match:
        return None
    return match.group("prefix"), int(match.group("frame"))


def build_pairs(dataset_root: Path):
    rgb_dir = dataset_root / "rgb"
    bbox_dir = dataset_root / "bounding_box_2d_tight"

    rgb_map = {}
    for p in tqdm(list(rgb_dir.glob("*.png")), desc="Indexing RGB files"):
        key = parse_name(p, "rgb")
        if key is not None:
            rgb_map[key] = p

    bbox_map = {}
    for p in tqdm(list(bbox_dir.glob("*.npy")), desc="Indexing bbox files"):
        key = parse_name(p, "bounding_box_2d_tight")
        if key is not None:
            bbox_map[key] = p

    common = sorted(set(rgb_map.keys()) & set(bbox_map.keys()))
    missing_rgb = len(set(bbox_map.keys()) - set(rgb_map.keys()))
    missing_bbox = len(set(rgb_map.keys()) - set(bbox_map.keys()))

    pairs = [(rgb_map[k], bbox_map[k]) for k in common]
    return pairs, missing_rgb, missing_bbox


def coco_categories(category_map):
    return [
        {"id": cat_id, "name": cat_name, "supercategory": "surgical_instrument"}
        for cat_id, cat_name in sorted(category_map.items())
    ]


def label_json_for_npy(npy_path: Path):
    # Replicator_03_bounding_box_2d_tight_4999.npy
    # -> Replicator_03_bounding_box_2d_tight_labels_4999.json
    m = re.match(r"^(?P<prefix>.+)_bounding_box_2d_tight_(?P<frame>\d+)\.npy$", npy_path.name)
    if not m:
        return None
    return npy_path.parent / f"{m.group('prefix')}_bounding_box_2d_tight_labels_{m.group('frame')}.json"


def load_semantic_names(label_json_path: Path):
    if label_json_path is None or (not label_json_path.exists()):
        return {}
    with label_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                if isinstance(v, dict):
                    if "class" in v:
                        name = v["class"]
                    elif "name" in v:
                        name = v["name"]
                    elif "label" in v:
                        name = v["label"]
                    else:
                        name = str(v)
                else:
                    name = v
                out[int(k)] = str(name)
            except Exception:
                continue
    return out


def convert_subset(pairs, category_map, images_root: Path, label_conversion: dict):
    images = []
    annotations = []
    ann_id = 1
    unknown_raw_names = defaultdict(int)
    missing_label_json = 0

    canonical_norm_to_cat_id = {
        normalize_label_name(name): cid for cid, name in category_map.items()
    }

    for image_id, (img_path, npy_path) in enumerate(
        tqdm(pairs, desc=f"Converting {images_root.name}", unit="img"), start=1
    ):
        with Image.open(img_path) as img:
            width, height = img.size

        images.append(
            {
                "id": image_id,
                "file_name": str(img_path.relative_to(images_root)).replace("\\", "/"),
                "width": width,
                "height": height,
            }
        )

        label_json_path = label_json_for_npy(npy_path)
        sid_to_name = load_semantic_names(label_json_path)
        if len(sid_to_name) == 0:
            missing_label_json += 1

        records = np.load(npy_path, allow_pickle=False)
        for rec in records:
            sid = int(rec["semanticId"])
            raw_name = sid_to_name.get(sid, None)
            if raw_name is None:
                unknown_raw_names[f"id:{sid}"] += 1
                continue

            raw_norm = normalize_label_name(raw_name)
            mapped_norm = label_conversion.get(raw_norm, raw_norm)
            cat_id = canonical_norm_to_cat_id.get(mapped_norm, None)
            if cat_id is None:
                unknown_raw_names[raw_name] += 1
                continue

            x_min = float(rec["x_min"])
            y_min = float(rec["y_min"])
            x_max = float(rec["x_max"])
            y_max = float(rec["y_max"])

            x_min = max(0.0, min(x_min, width - 1))
            y_min = max(0.0, min(y_min, height - 1))
            x_max = max(0.0, min(x_max, width - 1))
            y_max = max(0.0, min(y_max, height - 1))

            bw = max(0.0, x_max - x_min)
            bh = max(0.0, y_max - y_min)
            if bw <= 0 or bh <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(category_map),
    }, dict(unknown_raw_names), missing_label_json


def materialize_split(split_pairs, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    new_pairs = []

    for img_src, npy_src in tqdm(split_pairs, desc=f"Copying {split_dir.name}", unit="img"):
        img_dst = split_dir / img_src.name
        shutil.copy2(img_src, img_dst)
        # Keep raw npy source unchanged and do not copy to split folder.
        new_pairs.append((img_dst, npy_src))

    return new_pairs


def main():
    parser = argparse.ArgumentParser("Convert Isaac Sim dataset to COCO and split train/val.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./data/surgical_instrument"),
        help="Dataset root containing rgb/ and bounding_box_2d_tight/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/surgical_instrument/annotations"),
        help="Output directory for train.json and val.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (0-1), default 0.8")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Root directory to create train/ and valid/ folders (default: <dataset-root>)",
    )
    parser.add_argument("--train-split-name", type=str, default="train", help="Folder name for train split")
    parser.add_argument("--val-split-name", type=str, default="valid", help="Folder name for validation split")
    parser.add_argument(
        "--write-split-coco",
        action="store_true",
        help="Write COCO annotations into split folders as _annotations.coco.json",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("config/label.json"),
        help="Canonical label map JSON (id->name), e.g. config/label.json",
    )
    parser.add_argument(
        "--label-conversion",
        type=Path,
        default=Path("config/label_conversion.json"),
        help="Raw-name to canonical-name conversion JSON",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    pairs, missing_rgb, missing_bbox = build_pairs(args.dataset_root)
    if not pairs:
        raise RuntimeError("No matched rgb/npy pairs were found.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * args.train_ratio)
    n_train = min(max(n_train, 1), n_total - 1)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    split_root = args.split_root or args.dataset_root
    train_img_dir = split_root / args.train_split_name
    val_img_dir = split_root / args.val_split_name

    train_pairs = materialize_split(train_pairs, train_img_dir)
    val_pairs = materialize_split(val_pairs, val_img_dir)

    category_map = load_label_map(args.label_map)
    label_conversion = load_label_conversion(args.label_conversion)

    train_coco, train_unknown, train_missing_label_json = convert_subset(
        train_pairs, category_map, train_img_dir, label_conversion
    )
    val_coco, val_unknown, val_missing_label_json = convert_subset(
        val_pairs, category_map, val_img_dir, label_conversion
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train.json"
    val_out = args.output_dir / "val.json"

    with train_out.open("w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    with val_out.open("w", encoding="utf-8") as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)

    if args.write_split_coco:
        train_split_out = train_img_dir / "_annotations.coco.json"
        val_split_out = val_img_dir / "_annotations.coco.json"
        with train_split_out.open("w", encoding="utf-8") as f:
            json.dump(train_coco, f, ensure_ascii=False, indent=2)
        with val_split_out.open("w", encoding="utf-8") as f:
            json.dump(val_coco, f, ensure_ascii=False, indent=2)

    print(f"Matched image/annotation pairs: {n_total}")
    print(f"Train/Val split: {len(train_pairs)}/{len(val_pairs)}")
    print(f"Missing rgb for npy files: {missing_rgb}")
    print(f"Missing npy for rgb files: {missing_bbox}")
    print(f"Train images copied to: {train_img_dir}")
    print(f"Val images copied to: {val_img_dir}")
    print(f"Missing per-frame label json (train/val): {train_missing_label_json}/{val_missing_label_json}")

    merged_unknown = defaultdict(int)
    for k, v in train_unknown.items():
        merged_unknown[k] += v
    for k, v in val_unknown.items():
        merged_unknown[k] += v
    if merged_unknown:
        print("Unknown/unmapped raw labels (top 20):")
        for k, v in sorted(merged_unknown.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {k}: {v}")

    print(f"Wrote: {train_out}")
    print(f"Wrote: {val_out}")
    if args.write_split_coco:
        print(f"Wrote: {train_img_dir / '_annotations.coco.json'}")
        print(f"Wrote: {val_img_dir / '_annotations.coco.json'}")


if __name__ == "__main__":
    main()
