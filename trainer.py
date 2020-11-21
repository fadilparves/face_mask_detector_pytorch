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

        self.conv_layer_2 = conv_layer_2 = Sequential(
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

    def forward(self, x: Tensor):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        out = out.view(-1, 2048)
        out = self.linear_layer(out)
        return out

    def prepare_data(self) -> None:
        self.mask_df = mask_df = pd.read_pickle(self.mask_df_path)
        train, validate = train_test_split(mask_df, test_size=0.3, random_state=0, stratify=mask_df['mask'])
        self.train_df = TorchDataset(train)
        self.validate_df = TorchDataset(validate)

        mask_num = mask_df[mask_df['mask']==1].shape[0]
        non_mask_num = mask_df[mask_df['mask']==0].shape[0]
        samples = [non_mask_num, mask_num]
        norm_weights = [1 - (x / sum(samples)) for x in samples]
        self.cross_entropy_loss = CrossEntropyLoss(weight=torch.tensor(norm_weights))

    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_df, batch_size=32, shuffle=True, num_workers=4)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.validate_df, batch_size=32, num_workers=4)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=0.00001)
    
    def training_steps(self, batch:dict, _batch_idx: int) -> Dict[str, Tensor]:
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)

        tensor_board_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensor_board_logs}

    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]:
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.cross_entropy_loss(outputs, labels)

        _, outputs = torch.max(outputs, dim=1)
        val_acc = accuracy_score(outputs.cpu(), labels.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensor_board_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensor_board_logs}

model = MaskDetectorTrainer(Path('./data/df_mask.pickle'))

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/weights.ckpt',
    save_weights_only=True,
    monitor='val_acc',
    mode='max'
)

trainer = Trainer(gpus= 1 if torch.cuda.is_available() else 0,
                  max_epochs=10,
                  checkpoint_callback=checkpoint_callback,
                  profiler=True)

trainer.fit(model)