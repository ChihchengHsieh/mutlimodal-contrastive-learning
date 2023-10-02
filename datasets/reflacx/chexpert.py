import os
import random
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
import torchvision.transforms as T

class REFLACXCheXpertDataset(torch.utils.data.Dataset):
    '''
    This class load REFLACX from  MIMIC-EYE dataset.
    '''

    def __init__(self,
                 df_path,
                 mimic_eye_path,
                 image_size,
                 split_str,
                 label_cols=r_constant.CHEXPERT_LABEL_COLS,
                 use_aug=False,
                 ):
        self.df_path = df_path
        self.mimic_eye_path = mimic_eye_path
        self.image_size = image_size
        self.label_cols = label_cols
        self.split_str = split_str
        self.use_aug = self.split_str == 'train' and use_aug

        self.df = pd.read_csv(self.df_path)

        if not self.split_str is None:
            self.df = self.df[self.df["split"] == self.split_str]

        self.__init_transforms()

        super().__init__()

    def __init_transforms(self):
        self.resize_transform = T.Compose(
            [
                T.Resize([self.image_size, self.image_size]),
            ]
        )

        if self.use_aug:
            self.aug_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        [self.image_size, self.image_size], scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(45),
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ]
            )

    def __get_paths(self, data):
        patient_id = data['subject_id']
        study_id = data['study_id']
        dicom_id = data['dicom_id']
        image_path = os.path.join(
            self.mimic_eye_path, f"patient_{patient_id}", "CXR-JPG", f"s{study_id}", f"{dicom_id}.jpg",
        )
        return image_path

    def __len__(self):
        return len(self.df)
    
    def __prepare_xray(self, data):
        image_path = self.__get_paths(data)
        xray = read_image(image_path).repeat(3, 1, 1)/255
        if self.use_aug and random.random() <0.95: # 5% for direct resize.
            xray = self.aug_transform(xray)
            # print(f"Using Augmentation in {self.split_str}.")
            if self.split_str == 'test' or self.split_str == 'val':
                raise StopIteration(f"Shouldn't use Augmentation in {self.split_str}")
        else:
            xray = self.resize_transform(xray)

        return xray
    
    def __prepare_chexpert_label(self, data):
        return  torch.tensor(data[self.label_cols]) == 1

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # determine paths
        xray = self.__prepare_xray(data)
        label = self.__prepare_chexpert_label(data)

        return xray, label

    def lesion_to_idx(self, disease: str) -> int:
        if not disease in self.label_cols:
            raise Exception("This disease is not the label.")

        return self.label_cols.index(disease)
        

    def idx_to_lesion(self, idx: int) -> str:
        if idx >= len(self.label_cols):
            return f"exceed label range :{idx}"

        return self.label_cols[idx]
