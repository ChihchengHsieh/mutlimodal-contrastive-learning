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
    weights: str = "cl"  # [cl, ImageNet, RandomInit]
    cl_model_name: str = None
    trainable_backbone_layers: int = 5  # [0, 5]
    release_fixed_weights_after: int = None


@dataclass
class ResNetClassifierArgs:
    name: str = None
    weights: str = "cl"  # [cl, ImageNet, RandomInit]
    cl_model_name: str = None
    trainable_backbone_layers: int = 5  # [0, 5]
    release_fixed_weights_after: int = None


@dataclass
class DETRArgs:
    name: str = None
    weights: str = "cl"  # [cl, ImageNet, RandomInit]
    cl_model_name: str = None
    trainable_backbone_layers: int = 5  # [0, 5]
    release_fixed_weights_after: int = None

    hidden_dim: int = 256
    dilation: bool = False
    position_embedding: str = "sine"
    dropout: float = 0.1
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
    pre_norm: bool = False
    num_queries: int = 100
    aux_loss: bool = True
    set_cost_class: float = 1
    set_cost_bbox: float = 5
    set_cost_giou: float = 2
    giou_loss_coef: float = 2
    bbox_loss_coef: float = 5
    eos_coef: float = 0.1
