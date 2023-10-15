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

    def __init__(self,
                 temperature: float,
                 lambda_0: float = 0.5) -> None:
        super(CLIPLoss, self).__init__()

        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if lambda_0 > 1 or lambda_0 < 0:
            raise ValueError('lambda_0 must be a float between 0 and 1.')
        self.lambda_0 = lambda_0
        self.lambda_1 = 1-lambda_0

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
    def __init__(self,
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
        self.clinical_backbone =  clinical_backbone

        self.m1_pooler = self.__get_pooler(m1_pool)
        self.m2_pooler = self.__get_pooler(m2_pool)

        self.loss_fn = CLIPLoss(
            temperature=temperature, lambda_0=lambda_0
        )

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
            'cl-loss': loss,
        }

        return losses, logits


