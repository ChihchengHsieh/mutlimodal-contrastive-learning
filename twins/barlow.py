import torch
import torch.nn as nn

"""
Implementation of Barlow Twins (https://arxiv.org/abs/2103.03230), adapted for ease of use for experiments from
https://github.com/facebookresearch/barlowtwins, with some modifications using code from 
https://github.com/lucidrains/byol-pytorch
"""

from torchvision import transforms as T

# from transform_utils import *


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


class NetWrapper(nn.Module):
    # from https://github.com/lucidrains/byol-pytorch
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer
        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)

        return representation


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    """
    Adapted from https://github.com/facebookresearch/barlowtwins for arbitrary backbones, and arbitrary choice of which
    latent representation to use. Designed for models which can fit on a single GPU (though training can be parallelized
    across multiple as with any other model). Support for larger models can be done easily for individual use cases by
    by following PyTorch's model parallelism best practices.
    """

    def __init__(
        self,
        backbone,
        latent_id,
        projection_sizes,
        lambd,
        image_size,
        scale_factor=1,
        augment_fn=None,
        augment_fn2=None,
    ):
        """

        :param backbone: Model backbone
        :param latent_id: name (or index) of the layer to be fed to the projection MLP
        :param projection_sizes: size of the hidden layers in the projection MLP
        :param lambd: tradeoff function
        :param scale_factor: Factor to scale loss by, default is 1
        """
        super().__init__()

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

        self.backbone = backbone
        self.backbone = NetWrapper(self.backbone, latent_id)
        self.lambd = lambd
        self.scale_factor = scale_factor
        # projector
        sizes = projection_sizes
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, x):

        x1, x2 = self.augment1(x), self.augment2(x)

        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        z1 = self.projector(z1) # (B, D)
        z2 = self.projector(z2) # (B, D)

        # empirical cross-correlation matrix
        c = torch.mm(self.bn(z1).T, self.bn(z2)) #(DB*BD = DD)
        c.div_(z1.shape[0]) 

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = self.scale_factor * (on_diag + self.lambd * off_diag)
        return loss


if __name__ == "__main__":
    import torchvision

    model = torchvision.models.resnet18(zero_init_residual=True)
    proj = [512, 512, 512, 512]
    twins = BarlowTwins(model, "avgpool", proj, 0.5)
    inp1 = torch.rand(2, 3, 224, 224)
    inp2 = torch.rand(2, 3, 224, 224)
    outs = twins(inp1, inp2)
    # model = model_utils.extract_latent.LatentHook(model, ['avgpool'])
    # out, dicti = model(inp1)
    print(outs)
    # print(model)
