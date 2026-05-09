import argparse
import jsonlines
from tqdm import tqdm
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.slconfig import SLConfig

# this id_map is only for coco dataset which has 80 classes used for training but 90 categories in total.
# which change the start label -> 0
# {"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}

coco_id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6} # example for 6-class surgical dataset
coco_key_list=list(coco_id_map.keys())
coco_val_list=list(coco_id_map.values())

def coco_to_xyxy(bbox):
    x, y, width, height = bbox
    x1 = round(x, 2)
    y1 = round(y, 2)
    x2 = round(x + width, 2)
    y2 = round(y + height, 2)
    return [x1, y1, x2, y2]


def load_label_list_from_cfg(cfg_path):
    cfg = SLConfig.fromfile(cfg_path)
    if not hasattr(cfg, "label_list"):
        raise ValueError(f"'label_list' not found in cfg: {cfg_path}")
    return list(cfg.label_list)


def write_label_map_from_cfg(cfg_path, output_path):
    label_list = load_label_list_from_cfg(cfg_path)
    label_map = {str(i): name for i, name in enumerate(label_list)}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    return label_map


def coco2odvg(args):
    from pycocotools.coco import COCO
    coco = COCO(args.input)

    cats = coco.loadCats(coco.getCatIds())
    nms = {cat['id']:cat['name'] for cat in cats}
    metas = []

    use_coco_idmap = (args.idmap == 'coco2017')
    align_with_cfg = args.cfg is not None
    label_list = load_label_list_from_cfg(args.cfg) if align_with_cfg else None

    for img_id, img_info in tqdm(coco.imgs.items()):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        instance_list = []
        for ann_id in ann_ids:
            ann = coco.anns[ann_id]
            bbox = ann['bbox']
            bbox_xyxy = coco_to_xyxy(bbox)
            label = ann['category_id']
            category = nms[label]
            if use_coco_idmap:
                ind=coco_val_list.index(label)
                label_trans = coco_key_list[ind]
            else:
                label_trans = label
            if align_with_cfg:
                label_trans = int(label_trans) + args.label_offset
                if label_trans < 0 or label_trans >= len(label_list):
                    continue
                category = label_list[label_trans]
            instance_list.append({
                "bbox": bbox_xyxy,
                "label": label_trans,
                "category": category
                }
            )
        metas.append(
            {
                "filename": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "detection": {
                    "instances": instance_list
                }
            }
        )
    print("  == dump meta ...")
    with jsonlines.open(args.output, mode="w") as writer:
        writer.write_all(metas)
    print("  == done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("coco to odvg format.", add_help=True)
    parser.add_argument("--input", '-i', required=True, type=str, help="input list name")
    parser.add_argument("--output", '-o', required=True, type=str, help="output list name")
    parser.add_argument("--idmap", default='coco', type=str, help="if coco2017 use the coco2017 idmap, otherwise keep labels as is.")
    parser.add_argument("--cfg", type=str, default=None, help="cfg path with label_list to align labels/categories")
    parser.add_argument("--label-offset", type=int, default=-1, help="offset applied to COCO category_id when --cfg is used (default: -1 for 1-based -> 0-based)")
    parser.add_argument("--label-map-out", type=str, default=None, help="optional output path for label.json generated from cfg label_list")
    args = parser.parse_args()

    if args.label_map_out:
        if args.cfg is None:
            raise ValueError("--label-map-out requires --cfg")
        write_label_map_from_cfg(args.cfg, args.label_map_out)
    coco2odvg(args)
