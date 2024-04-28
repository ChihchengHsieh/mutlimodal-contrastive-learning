import torch
import torch.nn as nn

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


