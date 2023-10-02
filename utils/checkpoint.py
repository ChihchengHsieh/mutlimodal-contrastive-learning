
from collections import OrderedDict
from datetime import datetime
import os
import pickle
import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
import shutil

def get_pretrained_backbone_weights(cp, backbone_str):
    backbone_str = "xray_backbone."
    backbone_cp_dict = OrderedDict({})
    for k, v in cp['model'].items():
        if k.startswith(backbone_str):
            backbone_cp_dict.update({k.removeprefix(backbone_str): v})

    return backbone_cp_dict

def get_model_path(train_info, performance_dict):
    current_time_string = datetime.now().strftime("%m-%d-%Y %H-%M-%S")

    model_path = f"{train_info.config.training.name}_{train_info.config.model.name}"

    for k, v in performance_dict.items():
        model_path += f"_{k}_{v:.4f}"

    model_path += f"_epoch{train_info.epoch}_{current_time_string}"
    model_path = model_path.replace(":", "_").replace(".", "_")

    return model_path


def save_checkpoint(
    model_path: str,
    train_info,
    model: nn.Module,
    optimiser: Optimizer = None,
):

    saving_dict = {"model": model.state_dict()}
    if optimiser:
        saving_dict["optimiser"] = optimiser.state_dict()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", model_path), exist_ok=True)

    torch.save(
        saving_dict,
        os.path.join("checkpoints", model_path, "model"),
    )

    # saving the train_info.
    with open(
        os.path.join("checkpoints", model_path, "train_info.pkl"),
        "wb",
    ) as info_f:
        pickle.dump(train_info, info_f)

    train_info.final_model_path = model_path

    return train_info


def remove_existing_cp(previous_model: str):
    folder_path = os.path.join("checkpoints", previous_model)
    if not previous_model is None and os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Previous model: [{previous_model}] has been remove!!")


def load_checkpoints(model_path, device):
    folder_path = os.path.join("checkpoints", model_path)
    with open(os.path.join(folder_path, "train_info.pkl"), "rb") as f:
        train_info = pickle.load(f)

    cp = torch.load(
        os.path.join(folder_path, "model"), map_location=device
    )

    return train_info, cp
