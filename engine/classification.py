import math
import os
import sys
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import torch
from utils.checkpoint import get_pretrained_backbone_weights

from utils.loggers import MetricLogger, SmoothedValue
from tv_ref.utils import reduce_dict
from utils.tensor import nested_to_device
from torchvision.models import resnet18, ResNet18_Weights

cpu_device = torch.device("cpu")


def resnet_set_trainable_layers(
    model,
    trainable_layers,
):
    # select layers that wont be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(
            f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2",
                       "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return model

def load_backbone(config, device):
    if config.model.weights == 'cl':
        backbone = load_cl_pretrained(
            config.model.cl_model_name,
            device
        )
    elif config.model.weights == 'imagenet':
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        backbone = resnet18(weights=None)

    backbone = resnet_set_trainable_layers(
        backbone, config.model.trainable_backbone_layers if config.model.weights else 5)

    return backbone


def load_cl_pretrained(cl_model_name, device):
    # set load_part = "feature_extractors.xrays" to load the pretrained image backbone.

    backbone = resnet18(weights=None)


    # load weights into this backbone then apply fpn.
    cp = torch.load(
        os.path.join("checkpoints", cl_model_name, "model"), map_location=device
    )

    backbone_cp_dict = get_pretrained_backbone_weights(cp, "xray_backbone.")
    backbone.load_state_dict(backbone_cp_dict, strict=True)

    return backbone


class ClassificationEvaluator:
    def __init__(self) -> None:
        self.preds = []
        self.gts = []

    def update(self, outputs, targets):
        for o in outputs:
            self.preds.append(o.to(cpu_device).detach().numpy())

        for t in targets:
            self.gts.append(t.to(cpu_device).detach().numpy())

    def get_clf_score(self, clf_score, has_threshold=None):
        if has_threshold:
            return clf_score(
                np.array(self.gts).reshape(-1),
                (np.array(self.preds) > has_threshold).reshape(-1),
            )
        return clf_score(
            np.array(self.gts).reshape(-1), (np.array(self.preds)).reshape(-1)
        )

    def get_performance(
        self,
    ):
        return {
            "f1": self.get_clf_score(f1_score, has_threshold=0.5),
            "precision": self.get_clf_score(precision_score, has_threshold=0.5),
            "accuracy": self.get_clf_score(accuracy_score, has_threshold=0.5),
            "recall": self.get_clf_score(recall_score, has_threshold=0.5),
            "auc": self.get_clf_score(roc_auc_score, has_threshold=0.5),
        }
