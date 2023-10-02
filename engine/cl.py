
import math
import sys
import time
import numpy as np
import torch
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn

from utils.loggers import MetricLogger, SmoothedValue
from utils.dict import reduce_dict
from utils.tensor import nested_to_device
from utils.train import TrainingInfo
from torch.optim.optimizer import Optimizer
from copy import deepcopy


cpu_device = torch.device("cpu")


class ContrastiveLearningEvaluator(object):
    def __init__(self) -> None:
        self.logits = []
        self.labels = []

    def update(self, logits, targets):
        for l in logits:
            self.logits.append(l.to(cpu_device).detach())  # .numpy())

        for t in targets:
            self.labels.append(t.to(cpu_device).detach())  # .numpy())

    def get_accuracy(
        self,
    ):
        max_len = max([len(l) for l in self.logits])
        padded_logits = [
            F.pad(l, (0, max_len - len(l)), "constant", -np.inf) for l in self.logits
        ]
        padded_logits = torch.stack(padded_logits, dim=0)
        top1_acc_train = torchmetrics.Accuracy(
            task="multiclass", top_k=1, num_classes=padded_logits.shape[-1]
        )

        return top1_acc_train(
            padded_logits,
            torch.stack(self.labels, dim=0),
        ).item()

    def get_performance(self,):
        return {
            "accuracy": self.get_accuracy(),
        }


def train_one_epoch(model, optimiser, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for xrays, clinical in metric_logger.log_every(data_loader, print_freq, header):
        xrays = torch.stack(nested_to_device(xrays, device), axis=0)
        clinical = nested_to_device(clinical, device)
        loss_dict, _ = model(xrays, clinical)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimiser.zero_grad()
        losses.backward()
        optimiser.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimiser.param_groups[0]["lr"])

    return metric_logger


@torch.inference_mode()
def evaluate(model, data_loader, device, return_evaluator=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    # another evaluator
    if return_evaluator:
        evaluator = ContrastiveLearningEvaluator()

    for xrays, clinical in metric_logger.log_every(data_loader, 100, header):
        xrays = torch.stack(nested_to_device(xrays, device), axis=0)
        clinical = nested_to_device(clinical, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        loss_dict, outputs = model(xrays, clinical)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        outputs = outputs.to(cpu_device)
        model_time = time.time() - model_time

        if return_evaluator:
            evaluator.update(
                outputs,
                torch.arange(len(outputs), device="cpu"),
            )

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time,
                             loss=losses_reduced,
                             **loss_dict_reduced,
                             )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.set_num_threads(n_threads)

    if return_evaluator:
        return metric_logger, evaluator

    return metric_logger


