# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchvision import transforms

# import loader


def default(val, def_val):
    return def_val if val is None else val


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MoCo_v2(nn.Module):
    """
    from: https://github.com/AlexZaikin94/MoCo-v2/tree/master
    an implementation of MoCo v1 + v2

    MoCo v1: https://arxiv.org/abs/1911.05722
    MoCo v1: https://arxiv.org/abs/2003.04297
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = "resnet18",
        dim: int = 256,
        queue_size: int = 65536,
        batch_size: int = 128,
        momentum: float = 0.999,
        temperature: float = 0.07,
        bias: bool = True,
        moco: bool = False,
        clf_hyperparams: dict = dict(),
        seed: int = 42,
        mlp: bool = True,  # MoCo v2 improvement
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.dim = dim  # C
        self.queue_size = queue_size  # K
        self.batch_size = batch_size  # N
        self.momentum = momentum  # m
        self.temperature = temperature  # t
        self.bias = bias
        self.moco = moco
        self.clf_hyperparams = clf_hyperparams
        self.seed = seed
        if self.seed is not None:
            torch.random.manual_seed(self.seed)

        self.mlp = mlp
        self.args = args
        self.kwargs = kwargs

        # create the queue
        self.register_buffer(
            "queue",
            nn.functional.normalize(
                torch.randn(self.queue_size, self.dim, requires_grad=False), dim=1
            )
            / 10,
        )
        self.ptr = 0

        # use requested torchvision backbone as a base encoder
        self.base_encoder = vars(torchvision.models)[self.backbone]

        # create and init query encoder
        self.q_encoder = self.base_encoder(num_classes=self.dim)
        self.k_encoder = self.base_encoder(num_classes=self.dim)
        if self.mlp:
            self.q_encoder.fc = nn.Sequential(
                nn.Linear(
                    self.q_encoder.fc.weight.shape[1], self.q_encoder.fc.weight.shape[1]
                ),
                nn.ReLU(),
                nn.Linear(self.q_encoder.fc.weight.shape[1], self.dim, bias=self.bias),
            )
            self.k_encoder.fc = nn.Sequential(
                nn.Linear(
                    self.k_encoder.fc.weight.shape[1], self.k_encoder.fc.weight.shape[1]
                ),
                nn.ReLU(),
                nn.Linear(self.k_encoder.fc.weight.shape[1], self.dim, bias=self.bias),
            )

        # init key encoder with query encoder weights
        self.k_encoder.load_state_dict(self.q_encoder.state_dict())

        # freeze k_encoder params (for manual momentum update)
        for p_k in self.k_encoder.parameters():
            p_k.requires_grad = False

        # detach the fc layers for accessing the fc input encodings
        self.q_fc = self.q_encoder.fc
        self.k_fc = self.k_encoder.fc
        self.q_encoder.fc = Dummy()
        self.k_encoder.fc = Dummy()

    def end_moco_phase(self):
        """transition model to classification phase"""
        self.moco = False

        # delete non-necessary modules and freeze all weights
        del self.k_encoder
        del self.queue
        del self.ptr
        del self.k_fc
        for p in self.parameters():
            p.requires_grad = False

        # init new fc encoder layer
        self.q_encoder.fc = nn.Linear(
            self.q_fc[0].weight.shape[1], self.num_classes, bias=self.bias
        )
        self.q_encoder.fc.weight.data = torch.FloatTensor(self.clf.coef_)
        self.q_encoder.fc.bias.data = torch.FloatTensor(self.clf.intercept_)

        # del sklearn classifier and old mlp/fc layer
        del self.q_fc
        try:
            del self.clf
        except:
            pass

        # make sure new fc layer grad enables
        for p in self.q_encoder.fc.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters
        for p_q, p_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            p_k.data = p_k.data * self.momentum + (1.0 - self.momentum) * p_q.data
            p_k.requires_grad = False

        # update k_fc.parameters
        for p_q, p_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            p_k.data = p_k.data * self.momentum + (1.0 - self.momentum) * p_q.data
            p_k.requires_grad = False

    @torch.no_grad()
    def update_queue(self, k):
        """swap oldest batch with the current key batch and update ptr"""
        self.queue[self.ptr : self.ptr + self.batch_size, :] = k.detach().cpu()
        self.ptr = (self.ptr + self.batch_size) % self.queue_size
        self.queue.requires_grad = False

    def forward(self, *args, prints=False):
        if self.moco:
            return self.moco_forward(*args, prints=prints)
        else:
            return self.clf_forward(*args, prints=prints)

    def moco_forward(self, q, k, prints=False):
        """moco phase forward pass"""
        print("q in", q.shape) if prints else None
        print("k in", k.shape) if prints else None

        q_enc = self.q_encoder(q)  # queries: NxC
        q = self.q_fc(q_enc)
        q = nn.functional.normalize(q, dim=1)
        print("q_encoder(q)", q.shape) if prints else None

        with torch.no_grad():
            k = self.k_encoder(k)  # keys: NxC
            k = self.k_fc(k)
            k = nn.functional.normalize(k, dim=1)
        print("k_encoder(k)", k.shape) if prints else None

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        print("l_pos", l_pos.shape) if prints else None

        # negative logits: NxK
        print("self.queue", self.queue.shape) if prints else None
        l_neg = torch.einsum("nc,kc->nk", [q, self.queue.clone().detach()])
        print("l_neg", l_neg.shape) if prints else None

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        print("logits", logits.shape) if prints else None

        # contrastive loss labels, positive logits used as ground truth
        zeros = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        print("zeros", zeros.shape) if prints else None

        self.update_k_encoder_weights()
        self.update_queue(k)

        return q_enc.detach(), logits, zeros

    def clf_forward(self, x, prints=False):
        """clf phase forward pass"""
        print("x in", x.shape) if prints else None

        x = self.q_encoder(x)
        print("q_encoder(x)", x.shape) if prints else None

        return x

    def print_hyperparams(self):
        return f'{self.backbone}_dim{self.dim}_queue_size{self.queue_size}_batch_size{self.batch_size}_momentum{self.momentum}_temperature{self.temperature}_{"mlp" if self.mlp else "no_mlp"}'


class MoCoSingleGPU(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        image_size,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        augment_fn=None,
        augment_fn2=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoSingleGPU, self).__init__()

        DEFAULT_AUG = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.5
                ),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim, pretrained=True)
        # self.encoder_k = base_encoder(num_classes=dim, pretrained=True)
        self.encoder_q = base_encoder(pretrained=True)
        self.encoder_k = base_encoder(pretrained=True)

        self.encoder_q.fc = nn.Linear(self.encoder_q.fc.weight.shape[1], dim)
        self.encoder_k.fc = nn.Linear(self.encoder_k.fc.weight.shape[1], dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + (1.0 - self.m) * p_q.data
            p_k.requires_grad = False

        # fc is designed in encoder part.
        # # update k_fc.parameters
        # for p_q, p_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
        #     p_k.data = p_k.data * self.momentum + (1.0 - self.momentum) * p_q.data
        #     p_k.requires_grad = False

    @torch.no_grad()
    def update_queue(self, k):
        """swap oldest batch with the current key batch and update ptr"""

        batch_size = k.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # print(self.queue.shape)
        # print(k.shape)
        # self.queue[ptr : ptr + batch_size, :] = k.detach().cpu()
        self.queue[:, ptr : ptr + batch_size] = k.T
        ptr = (ptr + batch_size) % self.K
        self.queue.requires_grad = False
        self.queue_ptr[0] = ptr

    def forward(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        im_q, im_k = self.augment1(x), self.augment2(x)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(im_k)  # keys: NxC
            # k = self.k_fc(k)
            k = nn.functional.normalize(k, dim=1)
            # self._momentum_update_key_encoder()  # update the key encoder

            # # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # k = self.encoder_k(im_k)  # keys: NxC
            # k = nn.functional.normalize(k, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        # self._dequeue_and_enqueue(k)
        self.update_k_encoder_weights()
        self.update_queue(k)

        loss = self.criterion(logits, labels)
        return loss


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        augment_fn=None,
        augment_fn2=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        DEFAULT_AUG = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.5
                ),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        # self.encoder_q = base_encoder(num_classes=dim, pretrained=True)
        # self.encoder_k = base_encoder(num_classes=dim, pretrained=True)
        self.encoder_q = base_encoder(pretrained=True)
        self.encoder_k = base_encoder(pretrained=True)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        im_q, im_k = self.augment1(x), self.augment2(x)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
