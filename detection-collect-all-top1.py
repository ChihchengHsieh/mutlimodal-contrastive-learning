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
    # "our_improved_v4",
    # "our_improved_v4_without_auto",
    "our_improved_v8",
]

cmap = {
    "Enlarged cardiac silhouette": "yellow",
    "Atelectasis": "red",
    "Pleural abnormality": "orange",
    "Consolidation": "lightgreen",
    "Pulmonary edema": "dodgerblue",
}


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
    fig, ax = plt.subplots()

    plt.imshow(img, cmap="grey")
    width, height = img.size
    width_factor = width / img_size
    height_factor = height / img_size
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
    return fig


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
    fig, ax = plt.subplots()

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
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    # examples
    # parser.add_argume--lr",  4, type=float
    # parser.add_argument("--name", default="test", type=str, help="Name of the model")
    # parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--limited_lesion", default=False, action="store_true")
    parser.add_argument("--linear_eval", default=False, action="store_true")
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--top_k_score", default=None, type=int)

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
}


def main(args):

    # create the dataset to provide functions.
    test_dataset = REFLACXLesionDetectionDataset(
        image_size=args.image_size,
        split_str="test",
    )
    output_dir = Path(args.output_dir)

    args.tune_mode = "linear_eval" if args.linear_eval else "fine_tuned"

    for model_name in ["our_improved_v8"]:
        model_df = pd.read_csv(
            os.path.join(
                "detection_results",
                f"{model_name}_{args.tune_mode}_detection_preds.csv",
            )
        )
        for l in range(1, 6):
            l_name = test_dataset.idx_to_lesion(l)
            saving_folder = os.path.join(output_dir, model_name)
            os.makedirs(saving_folder, exist_ok=True)

            # only get the top-1
            sorted_lesion_df = model_df[model_df["label"] == l].sort_values(
                by="score", ascending=False
            )

            if args.top_k_score:
                sorted_lesion_df = sorted_lesion_df[: args.top_k_score]

            top_score_idx = 0
            for _, instance in sorted_lesion_df.iterrows():
                top_score_idx += 1
                # get the image
                dicom_id = os.path.basename(instance["image_path"]).split(".")[0]
                img = Image.open(instance["image_path"])

                x1, y1, x2, y2 = instance[["x1", "y1", "x2", "y2"]]
                # create the subplots
                fig, axes = plt.subplots(
                    ncols=(len(other_algs) + 2),
                    figsize=((len(other_algs) + 2) * 4, 4),
                )  # ground-truth <-other algorithms-> Our

                # put ground-truth at the first
                axes[0].set_title("GroundTruth")
                axes[0].imshow(img)

                plot_bb_on_ax(
                    axes[0],
                    img=img,
                    boxes=np.array(json.loads(instance["gt_boxes"])),
                    labels=np.array(json.loads(instance["gt_labels"])),
                    image_size=args.image_size,
                    idx_to_lesion_fn=test_dataset.idx_to_lesion,
                    limited_lesion=l if args.limited_lesion else None,
                )

                axes[-1].set_title("Our (Momentum)")
                # plot our at the last
                plot_bb_on_ax(
                    axes[-1],
                    img=img,
                    boxes=np.array([[x1, y1, x2, y2]]),
                    labels=np.array([instance["label"]]),
                    scores=np.array([instance["score"]]),
                    image_size=args.image_size,
                    idx_to_lesion_fn=test_dataset.idx_to_lesion,
                    limited_lesion=l if args.limited_lesion else None,
                )

                for ax, alg in zip(axes[1:-1], other_algs):
                    alg_df = pd.read_csv(
                        os.path.join(
                            "detection_results",
                            f"{alg}_{args.tune_mode}_detection_preds.csv",
                        )
                    )
                    # only keep the same dicom_id and the same lesion
                    alg_df = alg_df[alg_df["label"] == l]
                    this_img_bool = alg_df["image_path"].apply(
                        lambda x: contian_dicom_id(image_path=x, dicom_id=dicom_id)
                    )
                    alg_df = alg_df[this_img_bool]
                    alg_df = alg_df.sort_values(by="score", ascending=False)

                    ax.set_title(alg_to_name[alg])

                    if len(alg_df) == 0:
                        # plot directly
                        ax.imshow(img)
                    else:
                        alg_instance = alg_df.iloc[0]
                        alg_x1, alg_y1, alg_x2, alg_y2 = (
                            alg_instance["x1"],
                            alg_instance["y1"],
                            alg_instance["x2"],
                            alg_instance["y2"],
                        )

                        plot_bb_on_ax(
                            ax=ax,
                            img=img,
                            boxes=np.array([[alg_x1, alg_y1, alg_x2, alg_y2]]),
                            labels=np.array([alg_instance["label"]]),
                            scores=np.array([alg_instance["score"]]),
                            image_size=args.image_size,
                            idx_to_lesion_fn=test_dataset.idx_to_lesion,
                        )
                        # plt.plot()
                fig.tight_layout()
                fig.savefig(
                    os.path.join(
                        saving_folder,
                        f"{l_name}-{top_score_idx} [{dicom_id}].png",
                    )
                )
                plt.cla()
                plt.clf()
                plt.close("all")
                print(f"{l_name}-{top_score_idx} [{dicom_id}].png saved!")
                gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Collect all detection results scores from predictions",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
