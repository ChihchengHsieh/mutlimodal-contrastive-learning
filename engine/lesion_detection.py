from collections import OrderedDict
from copy import deepcopy
import torch
import os
import torch.nn as nn

from torchvision.models import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from utils.checkpoint import get_model_path, get_pretrained_backbone_weights, remove_existing_cp, save_checkpoint

def load_cl_pretrained(model, cl_model_name, trainable_backbone_layers):
    # set load_part = "feature_extractors.xrays" to load the pretrained image backbone.
    device = next(model.parameters()).device

    # load weights into this backbone then apply fpn.
    backbone = resnet50(weights=None)

    cp = torch.load(
        os.path.join("checkpoints", cl_model_name, "model"), map_location=device
    )

    backbone_cp_dict = get_pretrained_backbone_weights(cp, "xray_backbone.")

    backbone.load_state_dict(backbone_cp_dict, strict=True)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers)
    model.backbone = backbone

    return model


def get_ap_ar(
    evaluator,
):

    return {"ap": evaluator.stats[0], "ar": evaluator.stats[1]}


def check_best(
    train_info,
    model,
    optimiser,
    val_evaluator
):
    # Targeting the model with higher Average Recall and Average Precision.
    if train_info.val_losses[-1]['loss'] < train_info.best_val_loss:
        # do evaluation on test.
        previous_best_model = deepcopy(train_info.best_val_loss_model_path)
        performance_dict = get_ap_ar(
            val_evaluator.coco_eval["bbox"],
        )
        model_path = get_model_path(train_info, performance_dict)
        train_info.best_val_loss_model_path = model_path
        train_info.final_model_path = model_path
        train_info = save_checkpoint(
            model_path=model_path,
            train_info=train_info,
            model=model,
            optimiser=optimiser,
        )
        train_info.best_val_loss_model_path = train_info.final_model_path
        train_info.best_val_loss = train_info.val_losses[-1]['loss']
        if previous_best_model:
            remove_existing_cp(previous_best_model)

    return train_info


def end_train(
    train_info,
    model,
    optimiser,
    val_evaluator,
):

    train_info.timer.end_training()
    sec_took = train_info.timer.has_took_sec()
    print(
        f"| Training Done, start testing! | [{train_info.epoch}] Epochs Training time: [{sec_took}] seconds, Avg time / Epoch: [{sec_took/train_info.epoch}] seconds"
    )

    performance_dict = get_ap_ar(
        val_evaluator.coco_eval["bbox"],
    )

    model_path = get_model_path(train_info, performance_dict)
    train_info.final_model_path = model_path

    train_info = save_checkpoint(
        model_path=model_path,
        train_info=train_info,
        model=model,
        optimiser=optimiser,
    )

    print(train_info)

    return train_info
