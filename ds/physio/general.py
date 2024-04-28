import random
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torchvision.io import read_image


import torchvision.transforms as T
import os
import torch
import pandas as pd
import numpy as np

from PIL import Image

# training_clinical_mean_std = {
#     "age": {"mean": 62.924050632911396, "std": 18.486667896662354},
#     "temperature": {"mean": 98.08447784810126, "std": 2.7465209372955712},
#     "heartrate": {"mean": 85.95379746835444, "std": 18.967507646992733},
#     "resprate": {"mean": 18.15221518987342, "std": 2.6219004903965004},
#     "o2sat": {"mean": 97.85411392405064, "std": 2.6025150031174946},
#     "sbp": {"mean": 133.0685126582279, "std": 25.523304795054102},
#     "dbp": {"mean": 74.01107594936708, "std": 16.401336318103716},
#     "acuity": {"mean": 2.2610759493670884, "std": 0.7045539799670345},
#     "": {"mean": 4.0233658089377835, "std": 4.441189600688316},
# }

DEFAULT_MIMIC_CLINICAL_NUM_COLS: list[str] = [
    "age",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    # "pain",
    "acuity",
]

DEFAULT_MIMIC_CLINICAL_CAT_COLS: list[str] = ["gender"]


def get_number(x):
    try:
        return float(x)
    except:
        return None


def map_to_deviation_edge(x, deviation=1):
    return deviation if x else (-1) * deviation


class PhysioNetClinicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str = os.path.join("spreadsheets", "physio_clinical.csv"),
        physionet_path: str = "F:\\physionet.org",
        split_str: str = "train",
        clinical_numerical_cols: str = DEFAULT_MIMIC_CLINICAL_NUM_COLS,
        clinical_categorical_cols: str = DEFAULT_MIMIC_CLINICAL_CAT_COLS,
        image_size: int = 128,
        normalise_clinical_num: bool = True,
        resize: bool = False,
        cat_ohe=False,
    ):
        self.df_path = df_path
        self.physionet_path = physionet_path
        self.clinical_numerical_cols = clinical_numerical_cols
        self.clinical_categorical_cols = clinical_categorical_cols
        self.image_size = image_size
        self.normalise_clinical_num = normalise_clinical_num
        self.split_str = split_str
        self.resize = resize
        self.cat_ohe = cat_ohe
        # self.use_aug = self.split_str == "train" and use_aug

        self.df = pd.read_csv(self.df_path)
        # self.df["pain"] = self.df["pain"].apply(get_number)
        # self.df = self.df[~self.df["pain"].isna()]

        # before splitting the dataset, we first calculate the mean and std values on the training dataset.
        self.mean_std_map = {f: {} for f in self.clinical_numerical_cols}

        for f in self.clinical_numerical_cols:
            self.mean_std_map[f]["mean"] = self.df[f].mean()
            self.mean_std_map[f]["std"] = self.df[f].std()

        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.resize_transform = T.Compose(
            [
                T.Resize([self.image_size, self.image_size]),
            ]
        )

        self.__preprocess_clinical_df()
        super().__init__()

    def num_clinical_features(
        self,
    ):
        return len(self.clinical_numerical_cols) + len(self.clinical_categorical_cols)

    def __preprocess_clinical_df(
        self,
    ):
        if self.cat_ohe:
            self.dummy_cat_cols = []
            for col in self.clinical_categorical_cols:
                feature_dummies = pd.get_dummies(
                    self.df[col], prefix=col, drop_first=True
                )
                for col in feature_dummies.columns:
                    self.df[col] = feature_dummies[col].apply(
                        lambda x: map_to_deviation_edge(x)
                    )
                    self.dummy_cat_cols.append(col)
        else:
            self.encoders_map = {}
            for col in self.clinical_categorical_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders_map[col] = le

        if self.normalise_clinical_num:
            self.clinical_std_mean = {}
            for col in self.clinical_numerical_cols:
                # calculate mean and std
                mean = self.mean_std_map[col]["mean"]
                std = self.mean_std_map[col]["std"]
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
        # clinical_num = None
        # if (
        #     not self.clinical_numerical_cols is None
        #     and len(self.clinical_numerical_cols) > 0
        # ):
        return (
            torch.tensor(
                np.array(
                    data[self.clinical_numerical_cols + self.dummy_cat_cols],
                    dtype=float,
                )
            ).float()
            if self.cat_ohe
            else torch.tensor(
                np.array(
                    data[self.clinical_numerical_cols + self.clinical_categorical_cols],
                    dtype=float,
                )
            ).float()
        )
        # clinical_cat = None
        # if (
        #     not self.clinical_categorical_cols is None
        #     and len(self.clinical_categorical_cols) > 0
        # ):
        #     clinical_cat = {
        #         c: torch.tensor(np.array(data[c], dtype=int))
        #         for c in self.clinical_categorical_cols
        #     }

        # return {"cat": clinical_cat, "num": clinical_num}

    def __prepare_xray(self, data):
        image_path = self.__get_paths(data)
        # xray = Image.open(image_path)
        # if self.resize:
        xray = read_image(image_path).repeat(3, 1, 1) / 255
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


class PhysioNetImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str = os.path.join("spreadsheets", "physio_clinical.csv"),
        physionet_path: str = "F:\\physionet.org",
        split_str: str = "train",
        image_size: int = 128,
    ):
        self.df_path = df_path
        self.physionet_path = physionet_path
        self.image_size = image_size
        self.split_str = split_str
        self.df = pd.read_csv(self.df_path)

        # to match the size of clinical dataset.
        # self.df["pain"] = self.df["pain"].apply(get_number)
        # self.df = self.df[~self.df["pain"].isna()]

        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        self.resize_transform = T.Compose(
            [
                T.Resize([self.image_size, self.image_size]),
            ]
        )

        super().__init__()

    def get_img_path(self, data, version="2.0.0"):
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

    def prepare_xray(self, data):
        image_path = self.get_img_path(data)
        xray = read_image(image_path).repeat(3, 1, 1) / 255
        xray = self.resize_transform(xray)
        return xray

    def __getitem__(self, idx):
        data: pd.Series = self.df.iloc[idx]
        xray = self.prepare_xray(data)
        return xray

    def __len__(
        self,
    ):
        return len(self.df)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)
    rnd_gray = T.RandomGrayscale(p=0.2)
    color_distort = T.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


from PIL import ImageFilter


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


import torch.distributed as dist


@torch.no_grad()
def distributed_sinkhorn(out):
    world_size = -1
    epsilon = 0.05
    sinkhorn_iterations = 3

    Q = torch.exp(
        out / epsilon
    ).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


@torch.no_grad()
class GrayToRGB(object):
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        x = x.repeat(3, 1, 1) / 255
        return x


class PhysioNetMultiCropImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path: str = os.path.join("spreadsheets", "physio_clinical.csv"),
        physionet_path: str = "F:\\physionet.org",
        split_str: str = "train",
        image_size: int = 128,
    ):
        self.df_path = df_path
        self.physionet_path = physionet_path
        self.image_size = image_size
        self.split_str = split_str
        self.df = pd.read_csv(self.df_path)

        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        size_crops = [image_size]
        min_scale_crops = [0.14]
        max_scale_crops = [1]
        crops_for_assign = [0, 1]
        nmb_crops = [2]
        self.nmb_crops = nmb_crops
        self.crops_for_assign = crops_for_assign
        self.temperature = 0.1

        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = T.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                [
                    T.Compose(
                        [
                            randomresizedcrop,
                            T.RandomHorizontalFlip(p=0.5),
                            T.Compose(color_transform),
                            T.ToTensor(),
                            GrayToRGB(),
                            T.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans

        super().__init__()

    def get_img_path(self, data, version="2.0.0"):
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

    def prepare_xray(self, data):
        image_path = self.get_img_path(data)
        xray = Image.open(image_path)
        # xray = read_image(image_path).repeat(3, 1, 1) / 255
        multi_crops = list(map(lambda trans: trans(xray), self.trans))
        return multi_crops

    def __getitem__(self, idx):
        data: pd.Series = self.df.iloc[idx]
        multi_crops = self.prepare_xray(data)
        return multi_crops

    def __len__(
        self,
    ):
        return len(self.df)
