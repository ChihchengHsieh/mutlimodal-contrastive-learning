from dataclasses import dataclass, field


@dataclass
class MCLModelArgs:
    name: str = None
    clinical_emb_dims: int = 16
    clinical_out_channels: int = 1000
    cl_m1_pool: str = None
    cl_m2_pool: str = None
    cl_lambda_0: float = 0.5
    cl_temperature: float = 0.1
    cl_pj_pooled_dim: int = 1000
    cl_pj_embedding_dim: int = 1000
    cl_pj_dim: int = 128

@dataclass
class FasterRCNNArgs:
    name: str = None
    weights: str = "cl" #[cl, ImageNet, RandomInit]
    cl_model_name: str = None
    trainable_backbone_layers: int = 5 #[0, 5]

@dataclass
class ResNetClassifierArgs:
    name: str = None
    weights: str = "cl" #[cl, ImageNet, RandomInit]
    cl_model_name: str = None
    trainable_backbone_layers: int = 5 #[0, 5]

