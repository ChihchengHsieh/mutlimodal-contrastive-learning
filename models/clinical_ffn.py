
from collections import OrderedDict
from torch import nn
import torch


def chain_map(d: list[dict]) -> dict[str, list]:
    d_copy = {}
    for k in d[0].keys():
        d_copy[k] = []
    for t in d:
        for k, v in t.items():
            d_copy[k].append(v)
    return d_copy

class ClinicalFFN(nn.Module):
    # This extractor is used now mainly for contrastive learning (pre-training)
    """
    categorical_col_maps:{
        category_name: number of category
    }
    """

    def __init__(
        self,
        numerical_cols: list,
        categorical_col_maps: dict,
        embedding_dim: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numerical_cols = numerical_cols
        self.categorical_col_maps = categorical_col_maps
        self.out_channels = out_channels

        self.has_cat = len(categorical_col_maps) > 0
        self.has_num = len(self.numerical_cols) > 0

        self._init_embs()
        self._init_encoder()
    def _init_embs(self,):
        if self.has_cat:
            self.embs = nn.ModuleDict(
                {
                    k: nn.Embedding(v, self.embedding_dim)
                    for k, v in self.categorical_col_maps.items()
                }
            )

    def _init_encoder(self,):
        ## FNN   
        self.encoder_in_channels = len(self.numerical_cols) + (
            len(self.categorical_col_maps) * self.embedding_dim
        )
        # since we used embedding layer above, we altered to this one:
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.encoder_in_channels),
            nn.ReLU(),
            nn.Linear(self.encoder_in_channels, self.out_channels),
        )

    def emb_concat(self, x):

        cat_data = [x_i["cat"] for x_i in x]
        num_data = [x_i["num"] for x_i in x]
        cat_data = chain_map(cat_data)
        cat_data = {k: torch.stack(v, dim=0) for k, v in cat_data.items()}

        # num_data = None
        if self.has_num > 0:
            num_data = torch.stack(num_data)

        if self.has_cat:
            emb_out = OrderedDict(
                {k: self.embs[k](v) for k, v in cat_data.items()})
            emb_out_cat = torch.concat(list(emb_out.values()), axis=1)

            if self.has_num:
                tabular_input = torch.concat([num_data, emb_out_cat], dim=1)
            else:
                tabular_input = emb_out_cat
        else:
            tabular_input = num_data

        return tabular_input

    def forward(self, x):
        """
        [Input]
        {
                'cat' : categorical tabular data,
                'num' : numerical tabular data,
        }

        [Output] Tensor (B, self.out_channels)
        """

        tabular_input = self.emb_concat(x)
        output = self.encoder(tabular_input)

        return output
