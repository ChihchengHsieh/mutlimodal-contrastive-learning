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
import gc

#    --limited_lesion --linear_eval --output_dir detection_results/all-top1-row/  --top_k_score 5

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
    "our_improved_v8",  # momentum
]

cmap = {
    "Enlarged cardiac silhouette": "yellow",
    "Atelectasis": "red",
    "Pleural abnormality": "orange",
    "Consolidation": "lightgreen",
    "Pulmonary edema": "dodgerblue",
}


def plot_bb_on_ax(
    ax,
    img,
    boxes,
    labels,
    image_size,
    idx_to_lesion_fn,
    limited_lesion=None,
    scores=None,
):
    ax.imshow(img, cmap="grey")
    width, height = img.size
    width_factor = width / image_size
    height_factor = height / image_size
    if scores == None:
        scores = [None] * len(boxes)

    for bbox, label, score in zip(boxes, labels, scores):
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
            f"{disease}" if score is None else f"{disease}({score:.2f})",
            color="black",
            backgroundcolor=c,
        )


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Plot bounding boxes for healthy patients (False positive analysis).",
        add_help=False,
    )
    # examples
    # parser.add_argume--lr",  4, type=float
    # parser.add_argument("--name", default="test", type=str, help="Name of the model")
    # parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--linear_eval", default=False, action="store_true")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    return parser


def contian_dicom_id(image_path, dicom_id):
    return dicom_id in image_path


other_algs = [
    "supervised",
    "simsiam",
    "byol",
    "twins",
    "moco",
    "our_simclr",
    "swav",
]

alg_to_name = {
    "supervised": "Supervised",
    "simsiam": "SimSiam",
    "byol": "BYOL",
    "twins": "Barlow Twins",
    "moco": "MoCo",
    "our_simclr": "SimCLR",
    "swav": "SwAV",
    "our_improved_v4": "Our",
    "our_improved_v4_without_auto": "Our (No-AutoEncoder)",
    "our_improved_v8": "Our (Momentum)",  # momentum
}


def get_dicomid_from_path(image_path):
    return os.path.basename(image_path).split(".")[0]


def is_in_heath_ids(image_path, healthy_ids):
    dicom_id = get_dicomid_from_path(image_path)
    return dicom_id in healthy_ids


FP_count = {}


def main(args):

    # create the dataset to provide functions.
    test_dataset = REFLACXLesionDetectionDataset(
        image_size=args.image_size,
        split_str="test",
    )
    output_dir = Path(args.output_dir)

    args.tune_mode = "linear_eval" if args.linear_eval else "fine_tuned"

    reflacx_df = pd.read_csv("spreadsheets/reflacx_clinical.csv")

    all_healthy_dicom_ids = set(
        list(
            reflacx_df[
                (reflacx_df["No Finding_chexpert"] == 1.0)
                & (reflacx_df["No Finding_negbio"] == 1.0)
            ]["dicom_id"]
        )
    )

    for model_name in model_names:
        model_df = pd.read_csv(
            os.path.join(
                "detection_results",
                f"{model_name}_{args.tune_mode}_detection_preds.csv",
            )
        )

        healthy_df = model_df[
            (model_df["gt_boxes"] == "[]")
            & (
                model_df["image_path"].apply(
                    lambda x: is_in_heath_ids(x, all_healthy_dicom_ids)
                )
            )
        ]

        FP_count[model_name] = len(healthy_df)

        all_image_paths = set(list(healthy_df["image_path"]))
        saving_folder = os.path.join(output_dir, alg_to_name[model_name])
        os.makedirs(saving_folder, exist_ok=True)

        for image_path in all_image_paths:
            dicom_id = get_dicomid_from_path(image_path)
            img = Image.open(image_path).convert("RGB")
            image_healthy_df = healthy_df[healthy_df["image_path"] == image_path]
            fig, ax = plt.subplots()
            plot_bb_on_ax(
                ax,
                img=img,
                boxes=np.array(image_healthy_df[["x1", "y1", "x2", "y2"]]),
                labels=np.array(image_healthy_df["label"]),
                image_size=args.image_size,
                idx_to_lesion_fn=test_dataset.idx_to_lesion,
            )

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    saving_folder,
                    f"{model_name}-[{dicom_id}].png",
                )
            )

            fig, ax = plt.subplots()
            ax.imshow(img)
            fig.savefig(
                os.path.join(
                    saving_folder,
                    f"{model_name}-[{dicom_id}] (GT).png",
                )
            )

            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
            print(f"{model_name}-[{dicom_id}].png saved!")
    print(FP_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Plot bounding boxes for healthy patients (False positive analysis).",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# {'supervised': 554, 'simsiam': 712, 'byol': 528, 'twins': 467, 'moco': 378, 'our_simclr': 499, 'swav': 595, 'our_improved_v4': 502, 'our_improved_v4_without_auto': 505, 'our_improved_v8': 617}
