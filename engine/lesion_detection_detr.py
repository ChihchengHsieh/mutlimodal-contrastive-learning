from copy import deepcopy
import math
import sys
from typing import Iterable

# import torchvision
import torch
import tv_ref.utils as utils
# from tv_ref.coco_eval import CocoEvaluator
# from tv_ref.coco_utils import get_coco_api_from_dataset
from torchmetrics.detection import MeanAveragePrecision

cpu_device = torch.device("cpu")


def box_xyxy_to_cxcywh(x):
    if len(x) == 0:
        return x
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    # metric_logger.add_meter(
    #     "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    # )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    model.metric_logger = metric_logger

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        # images = images.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(images)


        # outputs = {
        #     k: v.float() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()
        # }

        for t in targets:
            # map to cxcywh
            t["boxes"] = box_xyxy_to_cxcywh(t["boxes"]) / data_loader.dataset.image_size
            if len(t["boxes"]) > 0:
                t['area'] = t['boxes'][:, 2] * t['boxes'][:, 3] 
            


        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {
        #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        # }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(
            loss=loss_value,
            # **loss_dict_reduced_scaled
            # loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        # metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return metric_logger


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types

import torch.nn.functional as F


@torch.no_grad()
def evaluate(
    model, criterion, data_loader, postprocessors, device, return_evaluator=False
):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter(
    #     "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    # )
    header = "Test:"

    # iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)

    if return_evaluator:
        evaluator = MeanAveragePrecision(iou_type="bbox", box_format="cxcywh")

    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = list(image.to(device) for image in images)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(images)

        # loss_targets = deepcopy(targets)

        for t in targets:
            # map to cxcywh
            t["boxes"] = box_xyxy_to_cxcywh(t["boxes"]) / data_loader.dataset.image_size
            if len(t["boxes"]) > 0:
                t['area'] = t['boxes'][:, 2] * t['boxes'][:, 3] 

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        # loss_dict_reduced_unscaled = {
        #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        # }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            # **loss_dict_reduced_scaled,
            # **loss_dict_reduced_unscaled,
        )
        # metric_logger.update(class_error=loss_dict_reduced["class_error"])

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()}
        #            for t in outputs]

        model.outputs = outputs

        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, out_bbox)
        ]
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors["bbox"](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target["image_id"]: output for target, output in zip(targets, results)}

        model.results = results
        model.targets = targets

        if return_evaluator:
            evaluator.update(results, targets)
            evaluator.cpu()

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name

        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # # if panoptic_evaluator is not None:
    # #     panoptic_evaluator.synchronize_between_processes()

    # # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if "bbox" in postprocessors.keys():
    #         stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    # if "segm" in postprocessors.keys():
    #     stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    if return_evaluator:
        return metric_logger, evaluator

    return (metric_logger,)

# @torch.no_grad()
# def evaluate(
#     model, criterion, data_loader, postprocessors, device, return_evaluator=False
# ):
#     model.eval()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     # metric_logger.add_meter(
#     #     "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
#     # )
#     header = "Test:"

#     # iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)

#     if return_evaluator:
#         coco_evaluator = CocoEvaluator(coco, iou_types)

#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

#     # panoptic_evaluator = None
#     # if 'panoptic' in postprocessors.keys():
#     #     panoptic_evaluator = PanopticEvaluator(
#     #         data_loader.dataset.ann_file,
#     #         data_loader.dataset.ann_folder,
#     #         output_dir=os.path.join(output_dir, "panoptic_eval"),
#     #     )

#     for images, targets in metric_logger.log_every(data_loader, 10, header):
#         images = list(image.to(device) for image in images)
#         targets = [
#             {
#                 k: v.to(device) if isinstance(v, torch.Tensor) else v
#                 for k, v in t.items()
#             }
#             for t in targets
#         ]

#         outputs = model(images)

#         loss_targets = deepcopy(targets)

#         for t in loss_targets:
#             # map to cxcywh
#             t["boxes"] = box_xyxy_to_cxcywh(t["boxes"]) / 128

#         loss_dict = criterion(outputs, loss_targets)
#         weight_dict = criterion.weight_dict

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {
#             k: v * weight_dict[k]
#             for k, v in loss_dict_reduced.items()
#             if k in weight_dict
#         }
#         # loss_dict_reduced_unscaled = {
#         #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
#         # }
#         metric_logger.update(
#             loss=sum(loss_dict_reduced_scaled.values()),
#             # **loss_dict_reduced_scaled,
#             # **loss_dict_reduced_unscaled,
#         )
#         # metric_logger.update(class_error=loss_dict_reduced["class_error"])

#         # outputs = [{k: v.to(cpu_device) for k, v in t.items()}
#         #            for t in outputs]

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results = postprocessors["bbox"](outputs, orig_target_sizes)
#         # if 'segm' in postprocessors.keys():
#         #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#         #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
#         res = {target["image_id"]: output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#         # if panoptic_evaluator is not None:
#         #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
#         #     for i, target in enumerate(targets):
#         #         image_id = target["image_id"].item()
#         #         file_name = f"{image_id:012d}.png"
#         #         res_pano[i]["image_id"] = image_id
#         #         res_pano[i]["file_name"] = file_name

#         #     panoptic_evaluator.update(res_pano)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     # if panoptic_evaluator is not None:
#     #     panoptic_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
#     # panoptic_res = None
#     # if panoptic_evaluator is not None:
#     #     panoptic_res = panoptic_evaluator.summarize()
#     # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     # if coco_evaluator is not None:
#     #     if "bbox" in postprocessors.keys():
#     #         stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
#     # if "segm" in postprocessors.keys():
#     #     stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
#     # if panoptic_res is not None:
#     #     stats['PQ_all'] = panoptic_res["All"]
#     #     stats['PQ_th'] = panoptic_res["Things"]
#     #     stats['PQ_st'] = panoptic_res["Stuff"]

#     if return_evaluator:
#         return metric_logger, coco_evaluator

#     return (metric_logger,)
