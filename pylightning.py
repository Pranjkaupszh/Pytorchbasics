import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer

INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES = 784, 500, 10
EPOCHS, BATCH_SIZE, LR = 2, 100, 0.001

#from lightning.ai documentation wrapped functions all
class LitNeuralNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        )
        self.val_losses = []

    def forward(self, x): 
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx): 
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx): 
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_losses).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        self.val_losses.clear()

    def configure_optimizers(self): 
        return torch.optim.Adam(self.parameters(), lr=LR)

    def train_dataloader(self): 
        return DataLoader(
            MNIST("./data", train=True, transform=transforms.ToTensor(), download=True),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self): 
        return DataLoader(
            MNIST("./data", train=False, transform=transforms.ToTensor()),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

if __name__ == '__main__':
    trainer = Trainer(
        max_epochs=EPOCHS,
        enable_progress_bar=True,
        logger=False,
        log_every_n_steps=1
    )
    trainer.fit(LitNeuralNet())
