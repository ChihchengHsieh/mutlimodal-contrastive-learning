import torch.nn as nn
import torch


# @torch.no_grad()
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor) for _ in range(1)]
#     # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#     output = torch.cat(tensors_gather, dim=0)
#     return output


class MoCoLoss(nn.Module):
    def __init__(self, temperature=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.T = temperature

    def forward(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum("nc,mc->nm", [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(
            N, dtype=torch.long
        ).cuda()  # + N * torch.distributed.get_rank()

        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
