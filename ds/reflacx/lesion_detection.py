import os
import torch


import pandas as pd

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes

# from torchvision import tv_tensors
# from torchvision.transforms.v2 import functional as F
import albumentations
import numpy as np
import torch
from . import constants as r_constant
from torchvision.ops.boxes import box_area


def box_xyxy_to_cxcywh(x):
    if len(x) == 0:
        return x
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class REFLACXLesionDetectionDataset(torch.utils.data.Dataset):
    """
    This class load REFLACX from  MIMIC-EYE dataset.
    """

    def __init__(
        self,
        df_path=os.path.join("spreadsheets", "reflacx.csv"),
        mimic_eye_path="F:\\mimic-eye",
        image_size=128,
        split_str="train",
        label_cols=r_constant.TOP5_LABEL_COLS,
        transform=None,
        cxcywh=False,
    ):
        self.df_path = df_path
        self.mimic_eye_path = mimic_eye_path
        self.image_size = image_size
        self.label_cols = label_cols
        self.split_str = split_str
        self.cxcywh = cxcywh

        self.df = pd.read_csv(self.df_path)

        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        self.__preprocess_label()
        self.__init_transform(transform)

        super().__init__()

    def __init_transform(self, transform):
        if not transform is None:
            self.transform = transform
        else:
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(self.image_size, self.image_size),
                    albumentations.HorizontalFlip(p=0.5),
                ],
                bbox_params=albumentations.BboxParams(
                    format="pascal_voc", label_fields=["label"]
                ),
            )

    def __get_paths(self, data):
        reflacx_id = data["id"]
        patient_id = data["subject_id"]
        study_id = data["study_id"]
        dicom_id = data["dicom_id"]
        image_path = os.path.join(
            self.mimic_eye_path,
            f"patient_{patient_id}",
            "CXR-JPG",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )
        bbox_path = os.path.join(
            self.mimic_eye_path,
            f"patient_{patient_id}",
            "REFLACX",
            "main_data",
            reflacx_id,
            "anomaly_location_ellipses.csv",
        )
        return image_path, bbox_path

    def __get_bb_df(self, bbox_path, img_height, img_width):
        bb_list = []
        bbox_df = pd.read_csv(bbox_path)
        for i, bb in bbox_df.iterrows():
            for l in [
                col for col in bb.keys() if not col in r_constant.DEFAULT_BOX_FIX_COLS
            ]:
                if bb[l] == True:
                    label = r_constant.DEFAULT_REPETITIVE_LABEL_REVERSED_MAP[l]
                    if label in self.label_cols:
                        xmax = np.clip(bb["xmax"], 0, img_width)
                        xmin = np.clip(bb["xmin"], 0, img_width)
                        ymax = np.clip(bb["ymax"], 0, img_height)
                        ymin = np.clip(bb["ymin"], 0, img_height)

                        # width = xmax-xmin
                        # height = ymax-ymin
                        # assert width >= 0, f"Width of BB should > 0, but got [{width}]"
                        # assert height >= 0, f"Height of BB should > 0, but got [{height}]"
                        # if width * height > 0:

                        bb_list.append(
                            {
                                "x_min": xmin,
                                "y_min": ymin,
                                "x_max": xmax,
                                "y_max": ymax,
                                "label": self.lesion_to_idx(label),
                            }
                        )

        return pd.DataFrame(
            bb_list, columns=["x_min", "y_min", "x_max", "y_max", "label"]
        )

    def __get_bb_label(self, bbox_path, img_height, img_width):
        bb_df = self.__get_bb_df(bbox_path, img_height, img_width)
        bbox = torch.tensor(
            np.array(bb_df[["x_min", "y_min", "x_max", "y_max"]], dtype=float)
        )
        label = torch.tensor(np.array(bb_df["label"]).astype(int), dtype=torch.int64)

        return {
            "bbox": bbox,
            "label": label,
        }

    def __preprocess_label(
        self,
    ):
        self.df[r_constant.ALL_LABEL_COLS] = self.df[r_constant.ALL_LABEL_COLS].gt(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # determine paths
        image_path, bbox_path = self.__get_paths(data)

        xray = read_image(image_path)  # grey, c = 1, (C, H, W)
        img_height, img_width = xray.shape[1], xray.shape[2]

        bb_label = self.__get_bb_label(bbox_path, img_height, img_width)
        num_objs = len(bb_label["bbox"])

        transformed = self.transform(
            image=xray.repeat(3, 1, 1).permute(1, 2, 0).numpy(),
            bboxes=bb_label["bbox"],
            label=bb_label["label"],
        )
        xray = torch.tensor(transformed["image"]).permute(2, 0, 1) / 255
        boxes = torch.tensor(transformed["bboxes"]).float()  # x1,y1,x2,y2

        target = {
            "image_id": idx,
            "boxes": boxes if len(boxes) > 0 else torch.zeros((0, 4)).float(),
            "labels": torch.tensor(transformed["label"], dtype=torch.int64),
            "area": box_area(boxes) if len(boxes) > 0 else torch.zeros((0, 4)).float(),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
            "orig_size": torch.tensor(
                [self.image_size, self.image_size], dtype=torch.int64
            ),
        }

        return xray, target

    def lesion_to_idx(self, disease: str) -> int:
        if not disease in self.label_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.label_cols.index(disease) + 1

    def idx_to_lesion(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.label_cols):
            return f"exceed label range :{idx}"

        return self.label_cols[idx - 1]
    
    def num_classes(
        self,
    ):
        return len(self.label_cols)
