# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from PIL import Image

from util.utils import to_device
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_iou

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator


def _target_boxes_to_xyxy_abs(tgt_boxes, tgt_size_hw):
    # tgt_boxes could be normalized cxcywh (common here) or already absolute xyxy.
    boxes = tgt_boxes.detach().cpu().float().clone()
    if boxes.numel() == 0:
        return boxes
    h = float(tgt_size_hw[0].item() if torch.is_tensor(tgt_size_hw[0]) else tgt_size_hw[0])
    w = float(tgt_size_hw[1].item() if torch.is_tensor(tgt_size_hw[1]) else tgt_size_hw[1])
    # Heuristic: if coordinates are in [0, 1.5], assume normalized cxcywh.
    if float(boxes.max()) <= 1.5:
        xyxy = boxes.clone()
        xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) * w
        xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) * h
        xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2.0) * w
        xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2.0) * h
    else:
        # Already absolute xyxy.
        xyxy = boxes
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, max(w - 1, 0))
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, max(h - 1, 0))
    return xyxy


def _compute_sample_outlier_score(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thr=0.5):
    # Higher score = worse sample (more errors).
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return 0.0
    if gt_boxes.numel() == 0:
        return float(pred_boxes.shape[0])
    if pred_boxes.numel() == 0:
        return float(gt_boxes.shape[0]) * 2.0

    ious = box_iou(gt_boxes, pred_boxes)  # [G, P]
    missed = 0
    fp = 0
    iou_penalty = 0.0

    # Missed GTs (same-label matching).
    for gi in range(gt_boxes.shape[0]):
        label_match = (pred_labels == gt_labels[gi])
        if label_match.any():
            best_iou = float(ious[gi, label_match].max().item())
            if best_iou < iou_thr:
                missed += 1
            iou_penalty += (1.0 - best_iou)
        else:
            missed += 1
            iou_penalty += 1.0

    # False positives (no same-label gt overlap).
    for pi in range(pred_boxes.shape[0]):
        label_match = (gt_labels == pred_labels[pi])
        if label_match.any():
            best_iou = float(ious[label_match, pi].max().item())
            if best_iou < iou_thr:
                fp += 1
        else:
            fp += 1

    return float(missed * 2.0 + fp + 0.2 * iou_penalty)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)

            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    outlier_topk = int(getattr(args, "visualize_eval_outliers", 0) or 0)
    bestmatch_topk = int(getattr(args, "visualize_eval_best_matches", 0) or 0)
    outlier_score_thr = float(getattr(args, "visualize_eval_outliers_score_thr", 0.30) or 0.30)
    outlier_records = []

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    coco_id_to_name = None
    if hasattr(base_ds, "cats") and isinstance(base_ds.cats, dict):
        coco_id_to_name = {int(k): str(v.get("name", k)) for k, v in base_ds.cats.items()}

    def _gt_label_to_name(lbl):
        # COCO-style targets store category_id directly.
        if coco_id_to_name is not None and lbl in coco_id_to_name:
            return coco_id_to_name[lbl]
        if 0 <= lbl < len(cat_list):
            return cat_list[lbl]
        if 1 <= lbl <= len(cat_list):
            return cat_list[lbl - 1]
        return str(lbl)

    def _normalize_gt_label_for_compare(lbl):
        # Non-COCO-eval path uses 0-based class indices in predictions
        # while COCO targets are commonly 1-based category ids.
        if not getattr(args, "use_coco_eval", False):
            if 1 <= lbl <= len(cat_list):
                return lbl - 1
        return lbl

    def _pred_label_to_name(lbl):
        # In this repo:
        # - use_coco_eval=False: labels are indices into cat_list.
        # - use_coco_eval=True: labels are COCO category ids after id_map in PostProcess.
        if getattr(args, "use_coco_eval", False):
            if coco_id_to_name is not None and lbl in coco_id_to_name:
                return coco_id_to_name[lbl]
        if 0 <= lbl < len(cat_list):
            return cat_list[lbl]
        if coco_id_to_name is not None and lbl in coco_id_to_name:
            return coco_id_to_name[lbl]
        if 1 <= lbl <= len(cat_list):
            return cat_list[lbl - 1]
        return str(lbl)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        vis_images = samples.tensors.detach().cpu()

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if outlier_topk > 0:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=vis_images.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=vis_images.dtype).view(3, 1, 1)
            for bi, (tgt, pred) in enumerate(zip(targets, results)):
                img_id = int(tgt["image_id"].item())
                oh, ow = int(tgt["orig_size"][0].item()), int(tgt["orig_size"][1].item())

                gt_boxes_xyxy = _target_boxes_to_xyxy_abs(tgt["boxes"], tgt["orig_size"])
                gt_labels = tgt["labels"].detach().cpu().long()

                pred_scores = pred["scores"].detach().cpu()
                keep = pred_scores > outlier_score_thr
                pred_boxes_xyxy = pred["boxes"].detach().cpu()[keep]
                pred_labels = pred["labels"].detach().cpu().long()[keep]
                pred_scores = pred_scores[keep]

                gt_labels_for_compare = gt_labels.clone()
                for gi in range(gt_labels_for_compare.numel()):
                    gt_labels_for_compare[gi] = _normalize_gt_label_for_compare(
                        int(gt_labels_for_compare[gi].item())
                    )

                score = _compute_sample_outlier_score(
                    gt_boxes_xyxy, gt_labels_for_compare, pred_boxes_xyxy, pred_labels
                )

                # Build display image at transformed size for readable overlays.
                base_img = (vis_images[bi] * std + mean).clamp(0, 1)
                base_img = (base_img * 255).to(torch.uint8)
                h, w = base_img.shape[-2], base_img.shape[-1]

                # Rescale orig-size absolute boxes to displayed size.
                sx = float(w) / float(max(ow, 1))
                sy = float(h) / float(max(oh, 1))
                gt_disp = gt_boxes_xyxy.clone()
                if gt_disp.numel() > 0:
                    gt_disp[:, [0, 2]] *= sx
                    gt_disp[:, [1, 3]] *= sy
                    gt_disp[:, [0, 2]] = gt_disp[:, [0, 2]].clamp(0, w - 1)
                    gt_disp[:, [1, 3]] = gt_disp[:, [1, 3]].clamp(0, h - 1)
                pred_disp = pred_boxes_xyxy.clone()
                if pred_disp.numel() > 0:
                    pred_disp[:, [0, 2]] *= sx
                    pred_disp[:, [1, 3]] *= sy
                    pred_disp[:, [0, 2]] = pred_disp[:, [0, 2]].clamp(0, w - 1)
                    pred_disp[:, [1, 3]] = pred_disp[:, [1, 3]].clamp(0, h - 1)

                gt_label_text = [_gt_label_to_name(int(x)) for x in gt_labels.tolist()]
                pred_label_text = [
                    f"{_pred_label_to_name(int(l))}:{float(s):.2f}"
                    for l, s in zip(pred_labels.tolist(), pred_scores.tolist())
                ]

                gt_img = base_img.clone()
                if gt_disp.numel() > 0:
                    gt_img = draw_bounding_boxes(gt_img, gt_disp, labels=gt_label_text, width=2, colors="green")
                pred_img = base_img.clone()
                if pred_disp.numel() > 0:
                    pred_img = draw_bounding_boxes(pred_img, pred_disp, labels=pred_label_text, width=2, colors="red")

                outlier_records.append(
                    {
                        "score": float(score),
                        "image_id": img_id,
                        "gt_count": int(gt_boxes_xyxy.shape[0]),
                        "gt_img": gt_img,
                        "pred_img": pred_img,
                    }
                )
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    if (outlier_topk > 0 or bestmatch_topk > 0) and utils.is_main_process():
        sorted_records = sorted(outlier_records, key=lambda x: x["score"], reverse=True)
        if outlier_topk > 0:
            out_dir = os.path.join(output_dir, "eval_outliers")
            os.makedirs(out_dir, exist_ok=True)
            top_outliers = sorted_records[:outlier_topk]
            for rank, rec in enumerate(top_outliers):
                gt_path = os.path.join(out_dir, f"rank{rank:03d}_img{rec['image_id']}_score{rec['score']:.3f}_gt.jpg")
                pred_path = os.path.join(out_dir, f"rank{rank:03d}_img{rec['image_id']}_score{rec['score']:.3f}_pred.jpg")
                Image.fromarray(rec["gt_img"].permute(1, 2, 0).numpy()).save(gt_path)
                Image.fromarray(rec["pred_img"].permute(1, 2, 0).numpy()).save(pred_path)
            if logger is not None:
                logger.info(f"Saved top-{len(top_outliers)} eval outlier visualizations to {out_dir}")

        if bestmatch_topk > 0:
            best_dir = os.path.join(output_dir, "eval_best_matches")
            os.makedirs(best_dir, exist_ok=True)
            best_candidates = [r for r in sorted_records if r.get("gt_count", 0) > 0]
            top_best = list(reversed(best_candidates[-bestmatch_topk:]))
            for rank, rec in enumerate(top_best):
                gt_path = os.path.join(best_dir, f"rank{rank:03d}_img{rec['image_id']}_score{rec['score']:.3f}_gt.jpg")
                pred_path = os.path.join(best_dir, f"rank{rank:03d}_img{rec['image_id']}_score{rec['score']:.3f}_pred.jpg")
                Image.fromarray(rec["gt_img"].permute(1, 2, 0).numpy()).save(gt_path)
                Image.fromarray(rec["pred_img"].permute(1, 2, 0).numpy()).save(pred_path)
            if logger is not None:
                logger.info(f"Saved top-{len(top_best)} eval best-match visualizations to {best_dir}")



    return stats, coco_evaluator


