from dataclasses import dataclass, field


@dataclass
class PhysioNetClinicalDatasetArgs:
    image_size: int = 128
    clinical_num: list[str] = field(
        default_factory=lambda: [
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
    )
    clinical_cat: list[str] = field(default_factory=lambda: ["gender"])
    categorical_col_maps: dict[str, int] = field(
        default_factory=lambda: {
            "gender": 2,
        }
    )
    normalise_clinical_num: bool = True
    use_aug: bool = True


@dataclass
class REFLACXLesionDetectionDatasetArgs:
    image_size: int = 128
    label_cols: list[str] = field(
        default_factory=lambda: [
            # "Fibrosis",
            # "Quality issue",
            # "Wide mediastinum",
            # "Fracture",
            # "Airway wall thickening",

            ######################
            # "Hiatal hernia",
            # "Acute fracture",
            # "Interstitial lung disease",
            # "Enlarged hilum",
            # "Abnormal mediastinal contour",
            # "High lung volume / emphysema",
            # "Pneumothorax",
            # "Lung nodule or mass",
            # "Groundglass opacity",
            ######################
            "Pulmonary edema",
            "Enlarged cardiac silhouette",
            "Consolidation",
            "Atelectasis",
            "Pleural abnormality",
            # "Support devices",
        ]
    )


@dataclass
class REFLACXCheXpertDatasetArgs:
    image_size: int = 128
    label_cols: list[str] = field(
        default_factory=lambda: [
            "Atelectasis_chexpert",
            "Cardiomegaly_chexpert",
            "Consolidation_chexpert",
            "Edema_chexpert",
            "Enlarged Cardiomediastinum_chexpert",
            "Fracture_chexpert",
            "Lung Lesion_chexpert",
            "Lung Opacity_chexpert",
            "No Finding_chexpert",
            "Pleural Effusion_chexpert",
            "Pleural Other_chexpert",
            "Pneumonia_chexpert",
            "Pneumothorax_chexpert",
            "Support Devices_chexpert",
        ]
    )
