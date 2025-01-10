"""
Now we want to test the performance of models merging the simple model with the other models.
"""


import linearmerge.simple_model as simple_model

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

import copy

# Load the data
data = pd.read_parquet("data/merged_data.parquet")

# drop nan
data = data.dropna()

print("number of rows", len(data))

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


class AirbnbDataset(Dataset):
    """
    Dataset for the Airbnb data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

        # separate the data into categorical and numerical data 
        # (int for categorical, float for numerical)
        self.categorical_data = data.select_dtypes(include=["category", "int8"])
        self.numerical_data = data.select_dtypes(include=["float"])

        self.dim_numerical = (
            len(self.numerical_data.columns) - 1
        )  # we remove the target later
        self.embedding_nb_categories = [
            np.max(self.categorical_data[col]) + 1
            for col in self.categorical_data.columns
        ]

        # now in numerical data, remove the target (price)
        target = "price"
        self.numerical_data = self.numerical_data.drop(target, axis=1)

        self.target = self.data[target] / 1000.0

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
dataloader = DataLoader(
    train_dataset, batch_size=2048, shuffle=True, drop_last=True
)  # cutting the final batch (acceptable)
test_dataset = AirbnbDataset(test_data)
dataloader_test = DataLoader(
    test_dataset, batch_size=2048, shuffle=False, drop_last=True
)  # cutting the final batch (acceptable)

# now we want to import the two models that are in models/
import os
simple_models = os.listdir("models/")
simple_models = [model for model in simple_models if "simple_model" in model]

models_0_state_dict = torch.load(
    f"models/{simple_models[0]}"
)
models_1_state_dict = torch.load(
    f"models/{simple_models[1]}"
)


time_values = np.linspace(0, 1, 100)

results_all = []

for time_value in time_values:
    model = simple_model.SimpleModel(
        dim_numerical=train_dataset.dim_numerical,
        embedding_nb_categories=train_dataset.embedding_nb_categories,
        dim_projective=10,
        hidden_size=128,
        output_size=1,
    )

    print(model)

    state_dict_0 = copy.deepcopy(models_0_state_dict)
    state_dict_1 = copy.deepcopy(models_1_state_dict)

    for key in state_dict_0:
        state_dict_0[key] = (
            time_value * state_dict_0[key] + (1 - time_value) * state_dict_1[key]
        )

    model.load_state_dict(state_dict_0)

    print(model)

    # now we want to test the model
    test_dataset = AirbnbDataset(test_data)
    dataloader_test = DataLoader(
        test_dataset, batch_size=2048, shuffle=False, drop_last=True
    )  # cutting the final batch (acceptable)


    trainer = pl.Trainer(max_epochs=10)
    results = trainer.test(model, dataloader_test)
    print(results)

    results_all.append(results[0]["test_loss"])

# save the results
results_all = np.array(results_all)
np.save("results/merge_simple.npy", results_all)