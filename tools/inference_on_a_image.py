import argparse
import os
import json
import ast
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets.transforms as T
from main import build_model_main
from util.slconfig import SLConfig
from util.utils import clean_state_dict
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_path, model_args=None):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    resize_scales = [800]
    resize_max_size = 1333
    if model_args is not None:
        resize_scales = getattr(model_args, "data_aug_scales", resize_scales)
        resize_max_size = getattr(model_args, "data_aug_max_size", resize_max_size)
    resize_target = max(resize_scales)

    transform = T.Compose(
        [
            T.RandomResize([resize_target], max_size=resize_max_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model, _, _ = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        if isinstance(token_spans, str):
            token_spans = ast.literal_eval(token_spans)
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(caption),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")

    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")
    parser.add_argument(
        "--closed-set",
        action="store_true",
        help="force labels to be one of dot-separated prompt phrases (closed-set mode)",
    )
    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    model_args = SLConfig.fromfile(config_file)
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    token_spans = args.token_spans

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path, model_args=model_args)
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # Build token spans from dot-separated phrases when using closed-set mode.
    if args.closed_set and token_spans is None:
        parts = [p.strip() for p in text_prompt.split(".") if p.strip()]
        if len(parts) == 0:
            raise ValueError("--closed-set requires a non-empty dot-separated text prompt")
        lc_caption = text_prompt.lower()
        spans = []
        cursor = 0
        for phrase in parts:
            lc_phrase = phrase.lower()
            start = lc_caption.find(lc_phrase, cursor)
            if start < 0:
                start = lc_caption.find(lc_phrase)
            if start < 0:
                continue
            end = start + len(lc_phrase)
            spans.append([[start, end]])
            cursor = end
        if len(spans) == 0:
            raise ValueError("Failed to derive phrase spans from --text_prompt in --closed-set mode")
        token_spans = spans

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")


    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, cpu_only=args.cpu_only, token_spans=token_spans
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)

    # Save and print per-box confidence details.
    predictions = []
    for box, phrase in zip(boxes_filt.tolist(), pred_phrases):
        confidence = None
        label = phrase
        if "(" in phrase and phrase.endswith(")"):
            cut = phrase.rfind("(")
            label = phrase[:cut].strip()
            conf_str = phrase[cut + 1 : -1]
            try:
                confidence = float(conf_str)
            except ValueError:
                confidence = None
        predictions.append(
            {
                "label": label,
                "confidence": confidence,
                "bbox_xywh_normalized": [float(v) for v in box],
            }
        )
    pred_json_path = os.path.join(output_dir, "predictions.json")
    with open(pred_json_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print("\nPer-box predictions:")
    for idx, item in enumerate(predictions, start=1):
        print(f"{idx}. label={item['label']}, confidence={item['confidence']}, bbox={item['bbox_xywh_normalized']}")
    print(f"\n======================\n{save_path} saved.\nThe program runs successfully!")
