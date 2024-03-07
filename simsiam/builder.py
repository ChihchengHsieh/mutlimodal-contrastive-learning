# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision import transforms as T


def default(val, def_val):
    return def_val if val is None else val


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(
        self, base_encoder, image_size, dim=2048, pred_dim=512, augment_fn=None, augment_fn2=None
    ):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

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
        self.encoder = base_encoder(zero_init_residual=True, pretrained=True)

        self.encoder.fc = nn.Linear(self.encoder.fc.weight.shape[1], dim)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # first layer
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),  # second layer
            self.encoder.fc,
            nn.BatchNorm1d(dim, affine=False),
        )  # output layer

        self.encoder.fc[6].bias.requires_grad = (
            False  # hack: not use bias as it is followed by BN
        )

        # build a 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),  # hidden layer
            nn.Linear(pred_dim, dim),
        )  # output layer

        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, x):
        """
        Input:
            x1: image.
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        x1, x2 = self.augment1(x), self.augment2(x)

        # compute features for one view
        z1 = self.encoder(x1)  # NxC
        z2 = self.encoder(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        loss = (
            -(
                self.criterion(p1, z2.detach()).mean()
                + self.criterion(p2, z1.detach()).mean()
            )
            * 0.5
        )

        return loss
