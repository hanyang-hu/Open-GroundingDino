import argparse
import json
import jsonlines
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.slconfig import SLConfig


def load_label_list_from_cfg(cfg_path):
    cfg = SLConfig.fromfile(cfg_path)
    if not hasattr(cfg, "label_list"):
        raise ValueError(f"'label_list' not found in cfg: {cfg_path}")
    return list(cfg.label_list)


def main():
    parser = argparse.ArgumentParser("Remap existing ODVG labels without rerunning coco2odvg.")
    parser.add_argument("--input", "-i", required=True, type=str, help="input ODVG jsonl")
    parser.add_argument("--output", "-o", required=True, type=str, help="output ODVG jsonl")
    parser.add_argument("--cfg", required=True, type=str, help="cfg with label_list")
    parser.add_argument(
        "--label-offset",
        type=int,
        default=-1,
        help="new_label = old_label + offset (default: -1 for 1-based -> 0-based)",
    )
    parser.add_argument(
        "--label-map-out",
        type=str,
        default=None,
        help="optional path to write label.json from cfg label_list",
    )
    args = parser.parse_args()

    label_list = load_label_list_from_cfg(args.cfg)
    metas = []
    dropped = 0

    with jsonlines.open(args.input, "r") as reader:
        for meta in tqdm(reader, desc="Remapping ODVG labels"):
            instances = meta["detection"]["instances"]
            new_instances = []
            for ins in instances:
                new_label = int(ins["label"]) + args.label_offset
                if new_label < 0 or new_label >= len(label_list):
                    dropped += 1
                    continue
                ins["label"] = new_label
                ins["category"] = label_list[new_label]
                new_instances.append(ins)
            meta["detection"]["instances"] = new_instances
            metas.append(meta)

    with jsonlines.open(args.output, "w") as writer:
        writer.write_all(metas)

    if args.label_map_out:
        label_map = {str(i): name for i, name in enumerate(label_list)}
        with open(args.label_map_out, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.output}")
    print(f"Dropped instances due to label range: {dropped}")


if __name__ == "__main__":
    main()
