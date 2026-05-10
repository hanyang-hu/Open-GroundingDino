import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_coco(ann_path: Path):
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    ann_by_image = {}
    for ann in data["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    return images, categories, ann_by_image


def draw_sample(image_path: Path, anns, categories, out_path: Path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for ann in anns:
        x, y, w, h = ann["bbox"]
        x1, y1 = x + w, y + h
        color = (0, 255, 0)
        draw.rectangle([x, y, x1, y1], outline=color, width=2)
        label = categories.get(ann["category_id"], str(ann["category_id"]))
        text = f"{label}"
        if hasattr(draw, "textbbox"):
            tb = draw.textbbox((x, y), text, font=font)
        else:
            tw, th = draw.textsize(text, font=font)
            tb = (x, y, x + tw, y + th)
        draw.rectangle(tb, fill=color)
        draw.text((x, y), text, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main():
    parser = argparse.ArgumentParser("Randomly sample COCO images and visualize bboxes.")
    parser.add_argument("--ann", required=True, type=Path, help="COCO annotation json path")
    parser.add_argument("--image-root", required=True, type=Path, help="Image root directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of random images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    images, categories, ann_by_image = load_coco(args.ann)
    image_ids = [img_id for img_id in images.keys() if img_id in ann_by_image]
    if len(image_ids) == 0:
        raise RuntimeError("No annotated images found in COCO file.")

    rng = random.Random(args.seed)
    n = min(args.num_samples, len(image_ids))
    sampled = rng.sample(image_ids, n)

    for idx, image_id in enumerate(sampled, start=1):
        img_info = images[image_id]
        img_path = args.image_root / img_info["file_name"]
        if not img_path.exists():
            # fallback if file_name includes subdirs or different relative root behavior
            img_path = args.image_root / Path(img_info["file_name"]).name
        if not img_path.exists():
            print(f"[WARN] Missing image: {img_info['file_name']}")
            continue
        out_name = f"{idx:03d}_{Path(img_info['file_name']).name}"
        out_path = args.out_dir / out_name
        draw_sample(img_path, ann_by_image[image_id], categories, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
