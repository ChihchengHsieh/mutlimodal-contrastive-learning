from typing import List
from git import Optional
import torch, math
import torch.nn as nn

# import loader
from torchvision import transforms as T
import torch.nn.functional as F

from loss.moco import MoCoLoss
from loss.simclr import SimCLRLoss
from loss.clip import CLIPLoss
from lightly.models.modules import SimCLRProjectionHead


def default(val, def_val):
    return def_val if val is None else val


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
        self.norm = RMSNorm(dim, eps=1e-5)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x):
        return x + self.ffn(self.norm(x))


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.Sequential(
            *[MLPBlock(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.out_layer(self.layers(self.in_layer(x)))


class TabularDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm = RMSNorm(in_dim, eps=1e-5)
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.Sequential(
            *[MLPBlock(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.out_layer(self.layers(self.in_layer(self.norm(x))))


class AutoEncoderLoss_v3(nn.Module):
    """
    Without sigmoid for BCELoss
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, x, y):
        """
        x: (B, D)
        y: (B, D)
        """
        return self.mse(x, y)


class TabAutoEncoder(nn.Module):
    def __init__(
        self,
        n_tab_layers,
        in_dim,
        hidden_dim,
        representation_size,
        out_dim,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enc = TabularEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=representation_size,
            n_layers=n_tab_layers,
        )

        self.dec = TabularDecoder(
            in_dim=representation_size,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=n_tab_layers,
        )

    def forward(self, x):
        out = self.dec(self.enc(x))
        return out


from itertools import combinations


def get_prediction_head(projection_dim: int, mlp_pj_dim_multiplier: int):
    return nn.Sequential(
        nn.Linear(projection_dim, projection_dim * mlp_pj_dim_multiplier, bias=False),
        nn.BatchNorm1d(projection_dim * mlp_pj_dim_multiplier),
        nn.ReLU(inplace=True),  # hidden layer
        nn.Linear(projection_dim * mlp_pj_dim_multiplier, projection_dim),
    )


class OurImproved_v13(nn.Module):
    """
    From alt->improve

    change the tab_enc, tab_dec -> tab_auto (adding sigmoid for categorical data)

    What we now from the result of v1 and alt.

    1. let's use the same augmentation of image for all the algorithm.
    2. use the default project head

    ## v4:
    Mimicking alt version.


    ## v5:
    1. changed the loss function.
    2. hi3, ht3 using clip proj. -> if it's not better than v4, then in v6, just use the same projector for all.
    - In this version, we test the new loss function, and see if using clip-proj for i3 has the better performance.

    ## v6:
    all loss function use the same proj.
    - In this version, we test if using the same projector for simclr and clip is better.

    ## v7. (*)
    utilise the encoder in best way, so construct as many loss as possible. # the reason of doing this is to get of the most from the least forward process.
    - in this version, we then test out whether should we get more loss as possible.
    - by reducing the forward process, we then increase the training time.

    ## v8
    - add MoCo momentum encoder
    - Add the projectors back, since SimCLR proves better to have them.

    ## v9
    swapped to SimCLR loss.

    ## v10.
    - use augmentation on clip loss.  -> Not better than v4, then we need to remove the momentum.

    ## v11:
    - get momentum and simclr together.


    ## v12:
    what can be the best algorithm?


    ## v13:
    Introduce MoCo to CLIP, and use dummy cat cols instead, so the autoencoder don't need to sigmoid (AutoEncoderLoss v3)
    """

    def __init__(
        self,
        base_encoder,
        clinical_input_dim,
        image_size,
        representation_dim=512,
        projection_dim=128,
        n_tab_layers=5,
        augment_fn=None,
        augment_fn2=None,
        augment_fn3=None,
        tab_drop_p=0.3,
        momentum=0.99,
        mlp_pj_dim_multiplier=16,
        tab_auto_hidden_multiplier=4,
    ):
        super(OurImproved_v13, self).__init__()

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
        self.augment3 = default(augment_fn3, self.augment1)
        self.m = momentum
        # self.mlp_dim_multiplier = mlp_dim_multiplier
        # self.tab_auto_hidden_multiplier = tab_auto_hidden_multiplier

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs

        # base encoder side
        self.img_enc = base_encoder(
            # num_classes=dim,
            zero_init_residual=True,
            pretrained=True,
        )
        self.img_enc.fc = nn.Identity()
        self.img_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.img_pred = get_prediction_head(
            projection_dim=projection_dim,
            mlp_pj_dim_multiplier=mlp_pj_dim_multiplier,
        )

        # momentum side
        self.m_img_enc = base_encoder(
            zero_init_residual=True,
            pretrained=True,
        )
        self.m_img_enc.fc = nn.Identity()
        self.m_img_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        # tabular base-encoder side
        self.tab_auto = TabAutoEncoder(
            n_tab_layers=n_tab_layers,
            in_dim=clinical_input_dim,
            hidden_dim=representation_dim * tab_auto_hidden_multiplier,
            out_dim=clinical_input_dim,
            representation_size=representation_dim,
        )

        self.tab_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.tab_pred = get_prediction_head(
            projection_dim=projection_dim,
            mlp_pj_dim_multiplier=mlp_pj_dim_multiplier,
        )

        # tabular momentum side
        self.m_tab_auto = TabAutoEncoder(
            n_tab_layers=n_tab_layers,
            in_dim=clinical_input_dim,
            hidden_dim=representation_dim * tab_auto_hidden_multiplier,
            out_dim=clinical_input_dim,
            representation_size=representation_dim,
        )
        self.m_tab_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.clip_img_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )
        self.clip_img_pred = get_prediction_head(
            projection_dim=projection_dim,
            mlp_pj_dim_multiplier=mlp_pj_dim_multiplier,
        )

        # clip momentum for image
        # self.clip_m_img_enc = base_encoder(
        #     zero_init_residual=True,
        #     pretrained=True,
        # )

        self.clip_m_img_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.clip_tab_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.clip_tab_pred = get_prediction_head(
            projection_dim=projection_dim,
            mlp_pj_dim_multiplier=mlp_pj_dim_multiplier,
        )
        # clip momentum for image
        # self.clip_m_tab_auto = TabAutoEncoder(
        #     input_dim=clinical_input_dim,
        #     dim=dim,
        #     n_tab_layers=n_tab_layers,
        # )

        self.clip_m_tab_pj = SimCLRProjectionHead(
            representation_dim,
            projection_dim * mlp_pj_dim_multiplier,
            projection_dim,
        )

        self.cross_loss_fn = CLIPLoss(temperature=0.1, lambda_0=0.5)
        self.img_loss_fn = MoCoLoss()  # Compare to SimCLR?
        self.tab_loss_fn = MoCoLoss()
        self.clip_loss_fn = MoCoLoss()
        self.auto_loss_fn = AutoEncoderLoss_v3()
        self.tab_drop_p = tab_drop_p

    def _update_momentum_encoders(self, enc, m_enc, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(enc.parameters(), m_enc.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1.0 - m)

    def forward(self, img, tab):
        i1, i2, i3 = self.augment1(img), self.augment2(img), self.augment3(img)

        t1, t2, t3 = (
            self.tab_auto(F.dropout(tab, p=self.tab_drop_p)),
            self.tab_auto(F.dropout(tab, p=self.tab_drop_p)),
            self.tab_auto(F.dropout(tab, p=self.tab_drop_p)),
        )

        zqi1, zqi2, zqi3, zqio = (
            self.img_pred(self.img_pj(self.img_enc(i1))),
            self.img_pred(self.img_pj(self.img_enc(i2))),
            self.clip_img_pred(self.clip_img_pj(self.img_enc(i3))),
            self.clip_img_pred(self.clip_img_pj(self.img_enc(img))),
        )

        zqt1, zqt2, zqt3, zqto = (
            self.tab_pred(self.tab_pj(self.tab_auto.enc(t1))),
            self.tab_pred(self.tab_pj(self.tab_auto.enc(t2))),
            self.clip_img_pred(self.clip_tab_pj(self.tab_auto.enc(t3))),
            self.clip_img_pred(self.clip_tab_pj(self.tab_auto.enc(tab))),
        )

        with torch.no_grad():
            self._update_momentum_encoders(self.img_enc, self.m_img_enc, self.m)
            self._update_momentum_encoders(self.img_pj, self.m_img_pj, self.m)

            zki1, zki2 = (
                self.m_img_pj(self.m_img_enc(i1)),
                self.m_img_pj(self.m_img_enc(i2)),
            )

            self._update_momentum_encoders(self.tab_auto, self.m_tab_auto, self.m)
            self._update_momentum_encoders(self.tab_pj, self.m_tab_pj, self.m)

            zkt1, zkt2 = (
                self.m_tab_pj(self.m_tab_auto.enc(t1)),
                self.m_tab_pj(self.m_tab_auto.enc(t2)),
            )

            # as the clip_m_img_enc is copied from self.img_enc, which is the same as self.m_img_enc.
            # using self.m_img_enc will be the same.
            # self._update_momentum_encoders(self.img_enc, self.clip_m_img_enc, self.m)
            self._update_momentum_encoders(self.clip_img_pj, self.clip_m_img_pj, self.m)

            zki3, zkio = (
                self.clip_m_img_pj(self.m_img_enc(i3)),
                self.clip_m_img_pj(self.m_img_enc(img)),
            )

            # self._update_momentum_encoders(self.tab_auto, self.clip_m_tab_auto, self.m)
            self._update_momentum_encoders(self.clip_tab_pj, self.clip_m_tab_pj, self.m)

            zkt3, zkto = (
                self.clip_m_tab_pj(self.m_tab_auto.enc(t1)),
                self.clip_m_tab_pj(self.m_tab_auto.enc(t2)),
            )

        img_loss = self.img_loss_fn(zqi1, zki2) + self.img_loss_fn(
            zqi2, zki1
        )  # symmetrized
        tab_loss = self.tab_loss_fn(zqt1, zkt2) + self.tab_loss_fn(
            zqt2, zkt1
        )  # symmetrized

        aug_clip_loss = self.tab_loss_fn(zqi3, zkt3) + self.tab_loss_fn(zqt3, zki3)
        original_clip_loss = self.tab_loss_fn(zqio, zkto) + self.tab_loss_fn(zqto, zkio)

        auto_l = (
            self.auto_loss_fn(tab, t1)
            + self.auto_loss_fn(tab, t2)
            + self.auto_loss_fn(tab, t3)
        ) / 3

        loss = (img_loss + tab_loss + aug_clip_loss + original_clip_loss + auto_l) / 5

        return loss, {
            "img_loss": img_loss.item(),
            "tab_loss": tab_loss.item(),
            "aug_clip_loss": aug_clip_loss.item(),
            "original_clip_loss": original_clip_loss.item(),
            "auto_l": auto_l.item(),
        }

    def _get_auto_loss(self, original: torch.Tensor, augmented: List[torch.Tensor]):
        loss = 0
        for a in augmented:
            loss += self.auto_loss_fn(original, a)
        return loss / len(augmented)

    def _get_comb_loss(self, zs, loss_fn):
        loss = 0
        combs = list(combinations(zs, 2))
        for comb in combs:
            loss += loss_fn(comb[0], comb[1])
        return loss / len(combs)

    def _get_cross_loss(self, img_zs, tab_zs):
        loss = 0
        for img_z in img_zs:
            for tab_z in tab_zs:
                loss += self.cross_loss_fn(img_z, tab_z)
        return loss / (len(img_zs) * len(tab_zs))
