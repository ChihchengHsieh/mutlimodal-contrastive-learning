from git import Optional
import torch, math
import torch.nn as nn

# import loader
from torchvision import transforms as T
import torch.nn.functional as F


def default(val, def_val):
    return def_val if val is None else val


class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, x1, x2):
        x1, x2 = nn.functional.normalize(x1, dim=1), nn.functional.normalize(x2, dim=1)
        b, device = x1.shape[0], x1.device
        logits = x1 @ x2.t()
        logits = logits - logits.max(dim=-1, keepdim=True).values
        logits /= self.temperature
        return F.cross_entropy(logits, torch.arange(b, device=device))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):

        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):

        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ffn = FeedForward(dim, hidden_dim)
        self.norm = RMSNorm(dim, eps=1e-5)

    def forward(self, x):
        return x + self.ffn(self.norm(x))


from lightly.models.modules import SimCLRProjectionHead


class SimCLR(nn.Module):
    """
    From alt->improve

    change the tab_enc, tab_dec -> tab_auto (adding sigmoid for categorical data)

    What we now from the result of v1 and alt.

    1. let's use the same augmentation of image for all the algorithm.
    2. use the default project head



    """

    def __init__(
        self,
        base_encoder,
        image_size,
        dim=512,
        pred_dim=128,
        augment_fn=None,
        augment_fn2=None,
    ):
        super(SimCLR, self).__init__()

        DEFAULT_AUG = T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                T.RandomApply(
                    [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur((3, 3), [0.1, 2.0])], p=0.5),
                T.RandomHorizontalFlip(),
                # T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.img_enc = base_encoder(
            # num_classes=dim,
            zero_init_residual=True,
            pretrained=True,
        )

        self.img_enc.fc = nn.Linear(self.img_enc.fc.weight.shape[1], dim)

        self.img_pj = SimCLRProjectionHead(
            dim,
            dim,
            pred_dim,
        )

        self.img_loss = SimCLRLoss(temperature=0.1)

    def forward(self, img, tab):
        i1, i2 = (
            self.augment1(img),
            self.augment2(img),
        )

        hi1, hi2 = (
            self.img_enc(i1),
            self.img_enc(i2),
        )

        zi1, zi2 = (
            self.img_pj(hi1),
            self.img_pj(hi2),
        )

        loss = self.img_loss(zi1, zi2)

        return loss, {}
