"""
Simple script to train a model on the preprocess merged data
"""

import pandas as pd
import numpy as np
import torch
from typing import List
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightning as pl

# define the trainer
from lightning.pytorch.callbacks import EarlyStopping
import linearmerge.simple_model as simple_model

# Load the data
data = pd.read_parquet("data/merged_data.parquet")

# drop nan
data = data.dropna()

print("number of rows", len(data))

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


class AirbnbDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # random shuffl
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        # separate the data into categorical and numerical data (int for categorical, float for numerical)
        self.categorical_data = data.select_dtypes(include=["category", "int8"])
        self.numerical_data = data.select_dtypes(include=["float"])

        self.dim_numerical = len(self.numerical_data.columns) - 1 # we remove the target later
        self.embedding_nb_categories = [
            np.max(self.categorical_data[col]) + 1 for col in self.categorical_data.columns
        ]

        # now in numerical data, remove the target (price)
        target = "price"
        self.numerical_data = self.numerical_data.drop(target, axis=1)

        self.target = self.data[target] / 1000.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        result_dict = {
            "categorical_data": self.categorical_data.iloc[idx].values,
            "numerical_data": self.numerical_data.iloc[idx].values,
            "target": self.target.iloc[idx],
        }

        return result_dict


train_dataset = AirbnbDataset(train_data)
dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, drop_last=True) # cutting the final batch (acceptable)



test_dataset = AirbnbDataset(test_data)
dataloader_test = DataLoader(test_dataset, batch_size=2048, shuffle=False, drop_last=True) # cutting the final batch (acceptable)




torch.manual_seed(42)

# define the model
model = simple_model.SimpleModel(
    dim_numerical=train_dataset.dim_numerical,
    embedding_nb_categories=train_dataset.embedding_nb_categories,
    dim_projective=10,
    hidden_size=128,
    output_size=1,
)

import wandb
wandb.init(project="airbnb-price-prediction")
wandb_logger = pl.pytorch.loggers.wandb.WandbLogger(project="airbnb-price-prediction")

trainer = pl.Trainer(
    max_epochs=10, logger=wandb_logger,
    gradient_clip_val=1.0
)

# train the model
trainer.fit(model, dataloader, dataloader_test)

# save the model with a random hash
import uuid

torch.save(model.state_dict(), f"models/simple_model_{uuid.uuid4()}.pth")
