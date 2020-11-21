from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU, Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .torch_dataset import TorchDataset

class MaskDetectorTrainer(pl.LightningModule):
    def __init__(self, mask_df_path: Path=None):
        super(MaskDetectorTrainer, self).__init__()
        self.mask_df_path = mask_df_path

        self.conv_layer_1 = conv_layer_1 = Sequential(
            Conv2d(3, 32, kernel_size=(3,3), padding=(1,1)),
            ReLU(),
            MaxPool2d(kernel_size=(2,2))
        )

        self conv_layer_2 = conv_layer_2 = Sequential(
            Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            ReLU(),
            MaxPool2d(kernel_size=(2,2))
        )

        self.conv_layer_3 = conv_layer_3 = Sequential(
            Conv2d(64, 128, kernel_size=(3,3), padding=(1,1), stride=(3,3)),
            ReLU(),
            MaxPool2d(kernel_size=(2,2))
        )

        self.linear_layer = linear_layer = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )

        for sequential in [conv_layer_1, conv_layer_2, conv_layer_3, linear_layer]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)