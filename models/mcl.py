import torch
import torch.nn as nn
from lightly.models.modules import SimCLRProjectionHead


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

        return loss, logits


import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class ExternalAutoEncoderCL(nn.Module):
    def __init__(self, auto_enc, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_enc = resnet18(weights=None)
        self.tab_enc = nn.Transformer()
        self.auto_enc = auto_enc
        self.img_aug = T.Compose(
            [
                T.RandomResizedCrop(
                    [self.image_size, self.image_size], scale=(0.8, 1.0)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(45),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ]
        )

        self.simclr_img_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_hidden_dim,
            pj_dim,
        )

        self.simclr_tab_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_hidden_dim,
            pj_dim,
        )

        self.clip_img_pj = CLIPProjector()
        self.clip_tab_pj = CLIPProjector()

        self.ci, self.ct, self.cit = 1, 1, 1

    def random_drop(
        self,
    ):
        pass

    # s: learnable log logit scale
    def clip(self, zi, zt):
        N = len(zi)
        zi, zt = normalize(zi, zt)
        label = range(N)
        logit = torch.exp(s) * zi @ zt.T
        li = CrossEntropy(logit, label)
        lt = CrossEntropy(logit.T, label)
        loss = (li + lt) / 2
        return loss

    def simclr(self, z1, z2, tau):
        z1, z2 = normalize(z1, z2)
        label = range(N)
        mask = eye(N) * 1e9
        logit = z1 @ z2.T
        logit1 = z1 @ z1.T - mask
        logit2 = z2 @ z2.T - mask
        logit1 = cat(logit, logit1)
        logit2 = cat(logit.T, logit2)
        l1 = CrossEntropy(logit1 / tau)
        l2 = CrossEntropy(logit2 / tau)
        loss = (l1 + l2) / 2
        return loss

    def forward(self, img, tab):
        # aug
        i1, i2, i3 = self.img_aug(img), self.img_aug(img), self.img_aug(img)
        t1, t2, t3 = (
            self.auto_enc(self.random_drop(tab)),
            self.auto_enc(self.random_drop(tab)),
            self.auto_enc(self.random_drop(tab)),
        )

        hi1, hi2, hi3 = self.img_enc(i1), self.img_enc(i2), self.img_enc(i3)
        ht1, ht2, ht3 = self.tab_enc(t1), self.tab_enc(t2), self.tab_enc(t3)

        zi1, zi2, zi3 = (
            self.simclr_img_pj(hi1),
            self.simclr_img_pj(hi2),
            self.clip_img_pj(hi3),
        )
        zt1, zt2, zt3 = (
            self.simclr_tab_pj(ht1),
            self.simclr_tab_pj(ht2),
            self.clip_tab_pj(ht3),
        )

        li = self.ci * self.simclr(zi1, zi2)
        lt = self.ct * self.simclr(zt1, zt2)
        lit = self.cit * self.clip(zi3, zt3)

        loss = (li + lt + lit) / 3

        return loss


class InternalAutoEncoderCLSharedEnc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_enc = resnet18(weights=None)
        self.tab_enc = nn.Transformer()
        self.tab_dec = nn.Linear()
        self.img_aug = T.Compose(
            [
                T.RandomResizedCrop(
                    [self.image_size, self.image_size], scale=(0.8, 1.0)
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(45),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            ]
        )

        self.simclr_img_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_hidden_dim,
            pj_dim,
        )

        self.simclr_tab_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_hidden_dim,
            pj_dim,
        )

        self.clip_img_pj = CLIPProjector()
        self.clip_tab_pj = CLIPProjector()

        self.ci, self.ct, self.cit, self.cauto = 1, 1, 1, 1

        self.mse = nn.MSELoss()

    def random_drop(
        self,
    ):
        pass

    # s: learnable log logit scale
    def clip(self, zi, zt):
        N = len(zi)
        zi, zt = normalize(zi, zt)
        label = range(N)
        logit = torch.exp(s) * zi @ zt.T
        li = CrossEntropy(logit, label)
        lt = CrossEntropy(logit.T, label)
        loss = (li + lt) / 2
        return loss

    def simclr(self, z1, z2, tau):
        z1, z2 = normalize(z1, z2)
        label = range(N)
        mask = eye(N) * 1e9
        logit = z1 @ z2.T
        logit1 = z1 @ z1.T - mask
        logit2 = z2 @ z2.T - mask
        logit1 = cat(logit, logit1)
        logit2 = cat(logit.T, logit2)
        l1 = CrossEntropy(logit1 / tau)
        l2 = CrossEntropy(logit2 / tau)
        loss = (l1 + l2) / 2
        return loss

    def forward(self, img, tab):
        # aug
        i1, i2, i3 = self.img_aug(img), self.img_aug(img), self.img_aug(img)
        (
            t1,
            t2,
            t3,
        ) = (
            self.tab_dec(self.tab_enc(self.random_drop(tab))),
            self.tab_dec(self.tab_enc(self.random_drop(tab))),
            self.tab_dec(self.tab_enc(self.random_drop(tab))),
        )

        hi1, hi2, hi3 = self.img_enc(i1), self.img_enc(i2), self.img_enc(i3)
        ht1, ht2, ht3 = (
            self.tab_enc(t1),
            self.tab_enc(t2),
            self.tab_enc(t3),
        )

        zi1, zi2, zi3 = (
            self.simclr_img_pj(hi1),
            self.simclr_img_pj(hi2),
            self.clip_img_pj(hi3),
        )
        zt1, zt2, zt3 = (
            self.simclr_tab_pj(ht1),
            self.simclr_tab_pj(ht2),
            self.clip_tab_pj(ht3),
        )

        li = self.ci * self.simclr(zi1, zi2)
        lt = self.ct * self.simclr(zt1, zt2)
        lit = self.cit * self.clip(zi3, zt3)
        lauto = self.cauto * (self.mse(tab, t1) + self.mse(tab, t2) + self.mse(tab, t3))

        loss = (li + lt + lit + lauto) / 4

        return loss


class ContrastiveLearningLoss(nn.Module):
    """
    CLIP:   (Image -><- Tabular data)
    SLIP:   (Image -><_ augmented Image) + (Image -><- Tabular)
    DeCLIP: (Image -><- augmented Image) (Image -><- Text)
            (Text -> Masked Language Modeling) + ((Augmented) Image -><- (Augmented) Text) + (Embedding text Clustering (Nearest-Neighbor)

    Our: Auto-Encoder for augmenting Clinical Data.
    Quest regarding design:
        1. Should we train the auto-encoder at the same time?
        2.
    """

    def __init__(self, temperature: float, lambda_0: float = 0.5) -> None:
        super(ContrastiveLearningLoss, self).__init__()

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

        return loss, logits


class MultimodalContrastiveLearning(nn.Module):

    def __init__(
        self,
        xray_backbone,
        clinical_backbone,
        m1_pool="avg",
        m2_pool=None,
        lambda_0=0.5,
        temperature=0.1,
        pj_pooled_dim=512,
        pj_embedding_dim=512,
        pj_dim=128,
    ) -> None:

        super().__init__()
        self.xray_backbone = xray_backbone
        self.clinical_backbone = clinical_backbone

        self.m1_pooler = self.__get_pooler(m1_pool)
        self.m2_pooler = self.__get_pooler(m2_pool)

        self.loss_fn = CLIPLoss(temperature=temperature, lambda_0=lambda_0)

        self.m1_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_embedding_dim,
            pj_dim,
        )

        self.m2_pj = SimCLRProjectionHead(
            pj_pooled_dim,
            pj_embedding_dim,
            pj_dim,
        )

    def __get_pooler(self, pool):
        if pool is None:
            return None

        if pool.lower() == "avg":
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"pool type {pool} is not implemented.")

    def forward(self, xray, clinical):
        self.xray = xray
        self.clinical = clinical
        z_m1 = self.xray_backbone(xray)
        z_m2 = self.clinical_backbone(clinical)

        if self.m1_pooler:
            z_m1 = self.m1_pooler(z_m1)
        if self.m2_pooler:
            z_m2 = self.m2_pooler(z_m2)

        z_m1 = self.m1_pj(z_m1)
        z_m2 = self.m2_pj(z_m2)

        loss, logits = self.loss_fn(z_m1, z_m2)

        losses = {
            "cl-loss": loss,
        }

        return losses, logits
