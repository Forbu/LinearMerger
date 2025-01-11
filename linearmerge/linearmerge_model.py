import torch
import torch.nn as nn
import lightning as pl
from typing import List

from linearmerge.linearmerge import LinearMerger

import numpy as np


class LinearMergeModel(pl.LightningModule):
    """
    A model with 6 layers:
    - 1 input layer
    - 4 hidden layers
    - 1 output layer
    """

    def __init__(
        self,
        dim_numerical: int,
        embedding_nb_categories: List[int],
        dim_projective: int,
        hidden_size: int,
        output_size: int,
        permute_order=None,
    ):
        super(LinearMergeModel, self).__init__()

        # generate random permute order
        torch.manual_seed(42)

        in_dim = dim_numerical + len(embedding_nb_categories) * dim_projective
        out_dim = hidden_size

        self.fc1 = LinearMerger(in_dim, out_dim, permute_order_seed=46)

        self.fc2 = LinearMerger(hidden_size, hidden_size, permute_order_seed=79)
        self.fc3 = LinearMerger(hidden_size, hidden_size, permute_order_seed=90)

        self.fc6 = nn.Linear(hidden_size, output_size)

        # random seed but different every time
        random_seep = np.random.randint(0, 1000000)
        torch.manual_seed(random_seep)

        print(f"Random seed: {random_seep}")
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

        # batch norm for numerical data
        self.batch_norm_numerical = nn.BatchNorm1d(dim_numerical)

        # embedding for categorical data
        self.dict_embedding = nn.ModuleDict(
            {
                str(key): nn.Embedding(dim_cate + 1, dim_projective)
                for key, dim_cate in enumerate(embedding_nb_categories)
            }
        )

        self.activation = nn.GELU()

    def forward(self, numerical_data: torch.Tensor, categorical_data: torch.Tensor):
        # apply batch norm to numerical data
        numerical_data = self.batch_norm_numerical(numerical_data)

        nb_categories = len(categorical_data.T)

        categorical_data = torch.cat(
            [
                self.dict_embedding[str(cat_id)](categorical_data[:, cat_id] + 1)
                for cat_id in range(nb_categories)
            ],
            dim=1,
        )

        # concatenate numerical and categorical data
        x = torch.cat([numerical_data, categorical_data], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc6(x)
        return x

    def training_step(self, batch, batch_idx):
        numerical_data, categorical_data, target = (
            batch["numerical_data"],
            batch["categorical_data"],
            batch["target"],
        )

        numerical_data = numerical_data.float()
        categorical_data = categorical_data.long()
        target = target.float()

        output = self(numerical_data, categorical_data)
        loss = nn.MSELoss()(output, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        numerical_data, categorical_data, target = (
            batch["numerical_data"],
            batch["categorical_data"],
            batch["target"],
        )

        numerical_data = numerical_data.float()
        categorical_data = categorical_data.long()
        target = target.float()

        output = self(numerical_data, categorical_data)
        loss = nn.MSELoss()(output, target)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        numerical_data, categorical_data, target = (
            batch["numerical_data"],
            batch["categorical_data"],
            batch["target"],
        )

        numerical_data = numerical_data.float()
        categorical_data = categorical_data.long()
        target = target.float()

        output = self(numerical_data, categorical_data)
        loss = nn.MSELoss()(output, target)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
