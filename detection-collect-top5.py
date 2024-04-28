#! /usr/bin/env python

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from ds.reflacx.lesion_detection import REFLACXLesionDetectionDataset
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# python -u deta_train.py --output_dir logs/deta --inspecting --with_box_refine --two_stage --num_queries 200 --ffn_dim 2048 --dropout 0.0 --cls_loss_coef 1.0 --image_size 256 --assign_first_stage --assign_second_stage --epochs 50 --lr_drop 20 --lr 3e-3 --batch_size 32

model_names = [
    "supervised",
    "simsiam",
    "byol",
    "twins",
    "moco",
    "our_simclr",
    "swav",
    "our_improved_v4",
    "our_improved_v4_without_auto",
    "our_improved_v8",
]

def plot_gt_on_img(
    img,
    boxes,
    labels,
    idx_to_lesion_fn,
    img_size,
    cmap={
        "Enlarged cardiac silhouette": "yellow",
        "Atelectasis": "red",
        "Pleural abnormality": "orange",
        "Consolidation": "lightgreen",
        "Pulmonary edema": "dodgerblue",
    },
    limited_lesion=None,
):
    fig, ax = plt.subplots(
        dpi=512,
    )

    plt.imshow(img, cmap="grey")
    width, height = img.size
    width_factor = width / img_size
    height_factor = height / img_size

    for bbox, label in zip(
        boxes,
        labels,
    ):
        if limited_lesion and label != limited_lesion:
            continue

        disease = idx_to_lesion_fn(label)
        c = cmap[disease]
        ax.add_patch(
            Rectangle(
                (bbox[0] * width_factor, bbox[1] * height_factor),
                (bbox[2] - bbox[0]) * width_factor,
                (bbox[3] - bbox[1]) * height_factor,
                fill=False,
                color=c,
                linewidth=2,
            )
        )
        ax.text(
            bbox[0] * width_factor,
            bbox[1] * height_factor,
            f"{disease}",
            color="black",
            backgroundcolor=c,
        )
    return fig


def plot_bboxes_on_img(
    img: torch.tensor,
    boxes,
    labels,
    scores,
    idx_to_lesion_fn,
    img_size,
    cmap={
        "Enlarged cardiac silhouette": "yellow",
        "Atelectasis": "red",
        "Pleural abnormality": "orange",
        "Consolidation": "lightgreen",
        "Pulmonary edema": "dodgerblue",
    },
):
    fig, ax = plt.subplots(
        dpi=512,
    )

    plt.imshow(img, cmap="grey")
    width, height = img.size
    width_factor = width / img_size
    height_factor = height / img_size

    for bbox, label, score in zip(
        boxes,
        labels,
        scores,
    ):
        # bbox = box_cxcywh_to_xyxy(torch.tensor(bbox * img_size)).numpy()
        disease = idx_to_lesion_fn(label)
        c = cmap[disease]
        ax.add_patch(
            Rectangle(
                (bbox[0] * width_factor, bbox[1] * height_factor),
                (bbox[2] - bbox[0]) * width_factor,
                (bbox[3] - bbox[1]) * height_factor,
                fill=False,
                color=c,
                linewidth=2,
            )
        )
        ax.text(
            bbox[0] * width_factor,
            bbox[1] * height_factor,
            f"{disease}({score:.2f})",
            color="black",
            backgroundcolor=c,
        )
    return fig


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=True)
    # examples
    # parser.add_argument("--lr", default=2e-4, type=float)
    # parser.add_argument("--name", default="test", type=str, help="Name of the model")
    # parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--limited_lesion", default=False, action="store_true")
    parser.add_argument("--linear_eval", default=False, action="store_true")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--top_k_score", default=5, type=int)
    return parser


def main(args):

    # create the dataset to provide functions.
    test_dataset = REFLACXLesionDetectionDataset(
        image_size=args.image_size,
        split_str="test",
    )

    args.tune_mode = "linear_eval" if args.linear_eval else "fine_tuned"

    for model_name in model_names:
        model_df = pd.read_csv(
            os.path.join(
                "detection_results", f"{model_name}_{args.tune_mode}_detection_preds.csv"
            )
        )
        for l in range(1, 6):
            top_5_score = model_df[model_df["label"] == l].sort_values(
                by="score", ascending=False
            )[:args.top_k_score]
            l_name = test_dataset.idx_to_lesion(l)
            saving_folder = os.path.join("detection_results", model_name)
            os.makedirs(saving_folder, exist_ok=True)
            idx = 0
            for _, instance in top_5_score.iterrows():
                idx += 1
                x1, y1, x2, y2 = (
                    instance["x1"],
                    instance["y1"],
                    instance["x2"],
                    instance["y2"],
                )
                img = Image.open(instance["image_path"])
                fig = plot_bboxes_on_img(
                    img=img,
                    boxes=np.array([[x1, y1, x2, y2]]),
                    labels=np.array([instance["label"]]),
                    scores=np.array([instance["score"]]),
                    img_size=args.image_size,
                    idx_to_lesion_fn=test_dataset.idx_to_lesion,
                )
                fig.savefig(os.path.join(saving_folder, f"{l_name}-{idx}.png"))
                gt_fig = plot_gt_on_img(
                    img=img,
                    boxes=np.array(json.loads(instance["gt_boxes"])),
                    labels=np.array(json.loads(instance["gt_labels"])),
                    img_size=args.image_size,
                    idx_to_lesion_fn=test_dataset.idx_to_lesion,
                    limited_lesion=l if args.limited_lesion else None,
                )
                gt_fig.savefig(os.path.join(saving_folder, f"{l_name}-{idx} (GT).png"))
                plt.cla()
                plt.clf()
                plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Collect top-5 scores from predictions", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
