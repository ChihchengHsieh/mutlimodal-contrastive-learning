from dataclasses import dataclass, field


@dataclass
class MultimodalContrastiveLearningArgs:
    name: str = None
    learning_rate: float = 0.03
    sgd_momentum: float = 0.9
    batch_size: int = 128
    weight_decay: float = 1e-4
    cl_pj_dim: int = 128
    clinical_cat_emb_dim: int = 16
    early_stopping_patience: int = 10
    warmup_epoch: int = 0


@dataclass
class LesionDetectionArgs:
    name: str = None
    learning_rate: float = 1e-2
    sgd_momentum: float = 0.9
    batch_size: int = 16
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    warmup_epoch: int = 0


@dataclass
class ImageClassificationArgs:
    name: str = None
    learning_rate: float = 0.03
    sgd_momentum: float = 0.9
    batch_size: int = 128
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    warmup_epoch: int = 0
