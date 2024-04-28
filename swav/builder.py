from git import Optional
import torch, math
import torch.nn as nn

# import loader
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import ImageFilter


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
    def __init__(self, in_dim, dim, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_layer = nn.Linear(in_dim, dim)
        self.layers = nn.Sequential(*[MLPBlock(dim, dim) for _ in range(n_layers)])

    def forward(self, x):
        return self.layers(self.in_layer(x))


class TabularDecoder(nn.Module):
    def __init__(self, dim, n_layers, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential(*[MLPBlock(dim, dim) for _ in range(n_layers)])
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


class AutoEncoderLoss_v2(nn.Module):
    """
    Without sigmoid for BCELoss
    """

    def __init__(self, n_cat_features, c=1) -> None:
        super().__init__()
        self.n_cat_features = n_cat_features
        self.mse = nn.MSELoss(reduction="mean")
        self.bce = nn.BCELoss(reduction="mean")
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


class TabAutoEncoder(nn.Module):
    def __init__(
        self, input_dim=10, cat_dim=1, n_tab_layers=2, dim=512, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enc = TabularEncoder(
            in_dim=input_dim,
            dim=dim,
            n_layers=n_tab_layers,
        )

        self.dec = TabularDecoder(
            dim=dim,
            n_layers=n_tab_layers,
            out_dim=input_dim,
        )
        self.cat_dim = cat_dim

    def forward(self, x):
        out = self.dec(self.enc(x))
        out[:, -1] = torch.sigmoid(out[:, -self.cat_dim])
        return out


import torchvision.transforms as transforms
import numpy as np
import random
# import torch.distributed as dist


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


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
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


class Debugger():
    pass

class SwAV(nn.Module):
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
    ):
        super(SwAV, self).__init__()
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
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend(
                [
                    transforms.Compose(
                        [
                            randomresizedcrop,
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.Compose(color_transform),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ]
                    )
                ]
                * nmb_crops[i]
            )
        self.trans = trans
        output_dim = 128

        # DEFAULT_AUG = T.Compose(
        #     [
        #         T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        #         T.RandomApply(
        #             [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        #         ),
        #         T.RandomGrayscale(p=0.2),
        #         T.RandomApply([T.GaussianBlur((3, 3), [0.1, 2.0])], p=0.5),
        #         T.RandomHorizontalFlip(),
        #         # T.ToTensor(),
        #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ]
        # )
        # self.augment1 = default(augment_fn, DEFAULT_AUG)
        # self.augment2 = default(augment_fn2, self.augment1)

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.img_enc = base_encoder(
            # num_classes=dim,
            zero_init_residual=True,
            pretrained=True,
        )

        self.img_enc.fc = nn.Linear(self.img_enc.fc.weight.shape[1], dim)

        # self.tab_auto = TabAutoEncoder(
        #     input_dim=clinical_input_dim,
        #     dim=dim,
        #     n_tab_layers=n_tab_layers,
        # )

        self.img_pj = SimCLRProjectionHead(
            dim,
            dim,
            pred_dim,
        )

        self._norm_layer = nn.BatchNorm2d
        self.padding = nn.ConstantPad2d(1, 0.0)

        widen = 1
        width_per_group = 64  # change to 128? in our case?
        self.inplanes = width_per_group * widen

        replace_stride_with_dilation = [False, False, False]
        self.groups = 1

        self.base_width = width_per_group
        num_out_filters = width_per_group * widen

        self.l2norm = True

        nmb_prototypes = 3000
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(
        self,
    ):
        w = self.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward_head(self, x):
        x = self.img_pj(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        return x, self.prototypes(x)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1],
            0,
        )

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.img_enc(
                torch.cat(inputs[start_idx:end_idx]).cuda(non_blocking=True)
            )
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        embedding, output = self.forward_head(output)
        embedding = embedding.detach()
        bs = inputs[0].size(0)

        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id : bs * (crop_id + 1)].detach()

                # get assignments
                q = distributed_sinkhorn(out)[-bs:]
            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                x = output[bs * v : bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)
        return loss
