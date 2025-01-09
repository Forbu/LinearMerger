"""
Simple script to train a model on the preprocess merged data
"""

import pandas as pd
import torch
from typing import List
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lighthning as pl


# Load the data
data = pd.read_parquet("../data/merged_data.parquet")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

class AirbnbDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # separate the data into categorical and numerical data (int for categorical, float for numerical)
        self.categorical_data = data.select_dtypes(include=['category', 'int']).columns
        self.numerical_data = data.select_dtypes(include=['float']).columns
        
        self.dim_input = len(self.categorical_data) + len(self.numerical_data)

        # now in numerical data, remove the target (price)
        target = 'price'
        self.target = self.numerical_data.pop(target)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'categorical_data': self.categorical_data.iloc[idx],
            'numerical_data': self.numerical_data.iloc[idx],
            'target': self.target.iloc[idx]
        }
        
dataloader = DataLoader(AirbnbDataset(train_data), batch_size=2048, shuffle=True)

class SimpleModel(pl.LightningModule):
    """
    A model with 6 layers:
    - 1 input layer
    - 4 hidden layers
    - 1 output layer
    """
    def __init__(self, dim_numerical: int, dim_categorical: List[str], dim_projective: int, hidden_size: int, output_size: int):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear((dim_numerical + dim_categorical) * dim_projective, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
    
        # batch norm for numerical data
        self.batch_norm_numerical = nn.BatchNorm1d(dim_numerical)
        
        # embedding for categorical data
        self.dict_embedding = nn.ModuleDict({
            key: nn.Embedding(dim_cate, dim_projective) for key, dim_cate in enumerate(dim_categorical)
        })
    
    def forward(self, numerical_data: torch.Tensor, categorical_data: torch.Tensor):
        
        # apply batch norm to numerical data
        numerical_data = self.batch_norm_numerical(numerical_data)
        
        # apply embedding to categorical data
        categorical_data = torch.cat([self.dict_embedding[key](categorical_data[:, i]) for key, i in enumerate(categorical_data.T)], dim=1)
        
        # concatenate numerical and categorical data
        x = torch.cat([numerical_data, categorical_data], dim=1)
        
        x = torch.relu(self.fc1(x))
        x_saved = x.clone()
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x)) + x_saved
        x = self.fc6(x)
        return x

# define the model
model = SimpleModel(input_size=10, hidden_size=10, output_size=1)

# define the trainer
trainer = pl.Trainer(max_epochs=10)

# train the model
trainer.fit(model, dataloader)

# save the model with a random hash
import uuid
torch.save(model.state_dict(), f"simple_model_{uuid.uuid4()}.pth")
