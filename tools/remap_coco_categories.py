import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.slconfig import SLConfig


def label_list_from_cfg(cfg_path):
    cfg = SLConfig.fromfile(cfg_path)
    if not hasattr(cfg, "label_list"):
        raise ValueError(f"'label_list' not found in cfg: {cfg_path}")
    return list(cfg.label_list)


def remap_coco_categories(data, label_list, id_offset=1, supercategory="surgical_instrument"):
    new_categories = []
    for i, name in enumerate(label_list):
        new_categories.append(
            {
                "id": i + id_offset,
                "name": name,
                "supercategory": supercategory,
            }
        )
    data["categories"] = new_categories
    return data


def main():
    parser = argparse.ArgumentParser("Remap COCO categories using cfg.label_list.")
    parser.add_argument("--input", "-i", required=True, type=str, help="input COCO json")
    parser.add_argument("--output", "-o", required=True, type=str, help="output COCO json")
    parser.add_argument("--cfg", required=True, type=str, help="cfg with label_list")
    parser.add_argument("--id-offset", type=int, default=1, help="category id start (default: 1)")
    parser.add_argument("--supercategory", type=str, default="surgical_instrument", help="supercategory string")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    label_list = label_list_from_cfg(args.cfg)
    data = remap_coco_categories(data, label_list, id_offset=args.id_offset, supercategory=args.supercategory)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
