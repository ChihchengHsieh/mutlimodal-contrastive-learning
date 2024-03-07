import random
from sklearn.calibration import LabelEncoder
from torchvision.io import read_image


import torchvision.transforms as T
import os
import torch
import pandas as pd
import numpy as np

training_clinical_mean_std = {
    "age": {"mean": 62.924050632911396, "std": 18.486667896662354},
    "temperature": {"mean": 98.08447784810126, "std": 2.7465209372955712},
    "heartrate": {"mean": 85.95379746835444, "std": 18.967507646992733},
    "resprate": {"mean": 18.15221518987342, "std": 2.6219004903965004},
    "o2sat": {"mean": 97.85411392405064, "std": 2.6025150031174946},
    "sbp": {"mean": 133.0685126582279, "std": 25.523304795054102},
    "dbp": {"mean": 74.01107594936708, "std": 16.401336318103716},
    "acuity": {"mean": 2.2610759493670884, "std": 0.7045539799670345},
}


class PhysioNetClinicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str,
        physionet_path: str,
        split_str: str,
        clinical_numerical_cols: str,
        clinical_categorical_cols: str,
        image_size: int = 128,
        normalise_clinical_num: bool = True,
        use_aug: bool = False,
    ):
        self.df_path = df_path
        self.physionet_path = physionet_path
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.image_size = image_size
        self.normalise_clinical_num = normalise_clinical_num
        self.split_str = split_str
        self.use_aug = self.split_str == "train" and use_aug

        self.df = pd.read_csv(self.df_path)

        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.__init_transforms()

        self.__preprocess_clinical_df()

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
                        [self.image_size, self.image_size], scale=(0.8, 1.0)
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(45),
                    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ]
            )

    def __preprocess_clinical_df(
        self,
    ):
        self.encoders_map = {}

        # encode the categorical cols.
        for col in self.clinical_categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

        if self.normalise_clinical_num:
            self.clinical_std_mean = {}
            for col in self.clinical_numerical_cols:
                # calculate mean and std
                mean = training_clinical_mean_std[col]["mean"]
                std = training_clinical_mean_std[col]["std"]
                self.df[col] = (self.df[col] - mean) / std

    def __get_paths(self, data, version="2.0.0"):
        patient_id = data["subject_id"]
        study_id = data["study_id"]
        dicom_id = data["dicom_id"]
        image_path = os.path.join(
            self.physionet_path,
            "files",
            "mimic-cxr-jpg",
            version,
            "files",
            f"p{str(patient_id)[:2]}",
            f"p{patient_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )
        return image_path

    def __prepare_clinical(self, data):
        clinical_num = None
        if (
            not self.clinical_numerical_cols is None
            and len(self.clinical_numerical_cols) > 0
        ):
            clinical_num = torch.tensor(
                np.array(data[self.clinical_numerical_cols], dtype=float)
            ).float()

        clinical_cat = None
        if (
            not self.clinical_categorical_cols is None
            and len(self.clinical_categorical_cols) > 0
        ):
            clinical_cat = {
                c: torch.tensor(np.array(data[c], dtype=int))
                for c in self.clinical_categorical_cols
            }

        return {"cat": clinical_cat, "num": clinical_num}

    def __prepare_xray(self, data):
        image_path = self.__get_paths(data)
        xray = read_image(image_path).repeat(3, 1, 1) / 255
        if self.use_aug and random.random() < 0.95:  # 5% for direct resize.
            xray = self.aug_transform(xray)
            # print(f"Using Augmentation in {self.split_str}.")
            if self.split_str == "test" or self.split_str == "val":
                raise StopIteration(f"Shouldn't use Augmentation in {self.split_str}")
        else:
            xray = self.resize_transform(xray)

        return xray

    def __getitem__(self, idx):
        data: pd.Series = self.df.iloc[idx]

        xray = self.__prepare_xray(data)
        clinical = self.__prepare_clinical(data)

        return xray, clinical

    def __len__(
        self,
    ):
        return len(self.df)


class PhysioNeImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clinical_numerical_cols: str,
        clinical_categorical_cols: str,
        df_path: str = os.path.join("spreadsheets", "physio_clinical.csv"),
        physionet_path: str = "F:\\physionet.org",
        split_str: str = "train",
        image_size: int = 128,
    ):
        self.df_path = df_path
        self.physionet_path = physionet_path
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.image_size = image_size
        self.split_str = split_str
        self.df = pd.read_csv(self.df_path)

        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.resize_transform = T.Compose(
            [
                T.Resize([self.image_size, self.image_size]),
            ]
        )

        super().__init__()

    def __get_img_path(self, data, version="2.0.0"):
        patient_id = data["subject_id"]
        study_id = data["study_id"]
        dicom_id = data["dicom_id"]
        image_path = os.path.join(
            self.physionet_path,
            "files",
            "mimic-cxr-jpg",
            version,
            "files",
            f"p{str(patient_id)[:2]}",
            f"p{patient_id}",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )
        return image_path

    def __prepare_xray(self, data):
        image_path = self.__get_img_path(data)
        xray = read_image(image_path).repeat(3, 1, 1) / 255
        xray = self.resize_transform(xray)
        return xray

    def __getitem__(self, idx):
        data: pd.Series = self.df.iloc[idx]
        xray = self.__prepare_xray(data)
        return xray

    def __len__(
        self,
    ):
        return len(self.df)
