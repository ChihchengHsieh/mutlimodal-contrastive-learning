from git import Optional
import torch, math
import torch.nn as nn

# import loader
from torchvision import transforms as T
import torch.nn.functional as F


def default(val, def_val):
    return def_val if val is None else val


class CLIPLoss(nn.Module):
    """
    Loss function for multimodal contrastive learning based off of the CLIP paper.
    Embeddings are taken, L2 normalized and dot product between modalities is calculated to generate a cosine
    similarity between all combinations of subjects in a cross-modal fashion. Tempered by temperature.
    Loss is calculated attempting to match each subject's embeddings between the modalities i.e. the diagonal.
    """

    def __init__(self, temperature: float, lambda_0: float = 0.5) -> None:
        super(CLIPLoss, self).__init__()

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError("lambda_0 must be a float between 0 and 1.")
        self.lambda_0 = lambda_0
        self.lambda_1 = 1 - lambda_0

    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        # normalize the embedding onto the unit hypersphere
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # logits = torch.matmul(out0, out1.T) * torch.exp(torch.tensor(self.temperature))
        # the cosine similarity function is (A \dot B) / ||A|| ||B||, since ||A|| = ||B|| = 1, we only need to calculate the molecular term.
        logits = torch.matmul(out0, out1.T) / self.temperature
        # Q: a list of [0, 1, 2, 3....] as labels? why? A: it's pairwise & symmetric, and only i=j should have the maximum likelihood.
        labels = torch.arange(len(out0), device=out0.device)

        loss_0 = self.lambda_0 * self.cross_entropy(logits, labels)
        loss_1 = self.lambda_1 * self.cross_entropy(logits.T, labels)
        loss = loss_0 + loss_1
        return loss  # , logits


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


class TabularEncoder(nn.Module):
    def __init__(self, in_dim, dim, hidden_dim, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_layer = nn.Linear(in_dim, dim)

        self.layers = nn.Sequential(
            *[MLPBlock(dim, hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x):
        return self.layers(self.in_layer(x))


class TabularDecoder(nn.Module):
    def __init__(self, dim, hidden_dim, n_layers, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(
            *[MLPBlock(dim, hidden_dim) for _ in range(n_layers)]
        )
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, x):
        return self.out_layer(self.layers(x))


class MixtralBLockSparseTop2MLP(nn.Module):
    def __init__(self, dim, intermediate_factor=3.5):
        super().__init__()
        self.dim = dim
        self.ffn_dim = math.ceil(dim * intermediate_factor)

        self.w1 = nn.Linear(self.dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.ffn_dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


from lightly.models.modules import SimCLRProjectionHead


class AutoEncoderLoss(nn.Module):
    def __init__(self, n_cat_features, c=1) -> None:
        super().__init__()
        self.n_cat_features = n_cat_features
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.c = c

    def forward(self, x, y):
        """
        x: (B, D)
        y: (B, D)
        """

        if self.n_cat_features > 0:
            num_x, num_y = x[:, : -self.n_cat_features], y[:, : -self.n_cat_features]
            cat_x, cat_y = x[:, -self.n_cat_features :], y[:, -self.n_cat_features :]
        else:
            num_x, num_y = x, y
            cat_x = cat_y = None

        if not cat_x is None and not cat_y is None:
            cat_loss = self.bce(cat_y, cat_x)
        else:
            cat_loss = 0

        if num_x.shape[1] > 0 and num_y.shape[1] > 0:
            num_loss = self.mse(num_y, num_x)
        else:
            num_loss = 0

        return self.c * cat_loss + num_loss


class OurImproved_v2(nn.Module):
    def __init__(
        self,
        base_encoder,
        clinical_input_dim,
        image_size,
        dim=512,
        pred_dim=128,
        n_tab_layers=5,
        n_cat_features=1,
        augment_fn=None,
        augment_fn2=None,
        augment_fn3=None,
        tab_drop_p=0.3,
        ci=1,
        ct=1,
        cit=1,
        c_auto=1,
    ):
        super(OurImproved_v2, self).__init__()

        DEFAULT_AUG = T.Compose(
            [
                T.RandomApply(
                    [T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8  # not strengthened
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.GaussianBlur((3, 3), [1.5, 1.5])], p=0.1),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomResizedCrop((image_size, image_size)),
            ]
        )
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)
        self.augment3 = default(augment_fn3, self.augment1)

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.img_enc = base_encoder(
            # num_classes=dim,
            zero_init_residual=True,
            pretrained=True,
        )

        self.img_enc.fc = nn.Linear(self.img_enc.fc.weight.shape[1], dim)

        # self.tab_emb = nn.Linear(clinical_input_dim, dim)
        self.tab_enc = TabularEncoder(
            in_dim=clinical_input_dim,
            dim=dim,
            hidden_dim=dim * 2,
            n_layers=n_tab_layers,
        )

        # self.tab_dec = TabularDecoder(
        #     dim=dim,
        #     hidden_dim=dim * 2,
        #     n_layers=n_tab_layers,
        #     out_dim=clinical_input_dim,
        # )

        # replace the projection function.
        self.img_pj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, pred_dim, bias=False),
        )

        self.tab_pj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, pred_dim, bias=False),
        )

        self.cross_img_pj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, pred_dim, bias=False),
        )

        self.cross_tab_pj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim, pred_dim, bias=False),
        )

        # self.img_pj = SimCLRProjectionHead(
        #     dim,
        #     dim,
        #     pred_dim,
        # )

        # self.tab_pj = SimCLRProjectionHead(
        #     dim,
        #     dim,
        #     pred_dim,
        # )

        # self.cross_img_pj = SimCLRProjectionHead(
        #     dim,
        #     dim,
        #     pred_dim,
        # )

        # self.cross_tab_pj = SimCLRProjectionHead(
        #     dim,
        #     dim,
        #     pred_dim,
        # )

        self.cross_loss = CLIPLoss(temperature=0.1, lambda_0=0.5)
        self.img_loss = SimCLRLoss(temperature=0.1)
        self.tab_loss = SimCLRLoss(temperature=0.1)
        self.auto_loss = AutoEncoderLoss(n_cat_features)

        self.ci, self.ct, self.cit, self.c_auto = ci, ct, cit, c_auto
        self.tab_drop_p = tab_drop_p

    def forward(self, img, tab):
        i1, i2 = (
            self.augment1(img),
            self.augment2(img),
        )

        t1, t2 = (F.dropout(tab, p=self.tab_drop_p), F.dropout(tab, p=self.tab_drop_p))

        hi1, hi2, hio = (
            self.img_enc(i1),
            self.img_enc(i2),
            self.img_enc(img),
        )

        ht1, ht2, hto = (
            self.tab_enc(t1),
            self.tab_enc(t2),
            self.tab_enc(tab),
        )

        zi1, zi2, zio = (
            self.img_pj(hi1),
            self.img_pj(hi2),
            self.cross_img_pj(hio),
        )
        zt1, zt2, zto = (
            self.tab_pj(ht1),
            self.tab_pj(ht2),
            self.cross_tab_pj(hto),
        )

        li = self.ci * self.img_loss(zi1, zi2)
        lt = self.ct * self.tab_loss(zt1, zt2)
        lc = self.cit * self.cross_loss(zio, zto)
        l_auto = self.c_auto * (self.auto_loss(tab, t1) + self.auto_loss(tab, t2))
        print(l_auto)
        loss = (li + lt + lc + l_auto) / 4

        return loss
