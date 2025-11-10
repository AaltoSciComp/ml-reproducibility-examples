import torch
from torch import nn
import lightning as L


class SimpleMLP(L.LightningModule):
    def __init__(self, hidden_size=20):
        # Init is done similar to nn.Module
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )
        # We specify loss function in the module as well
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # Forward is done similar to nn.Module
        return self.layers(x)

    def training_step(self, batch):
        # training_step-function specifies how data is fed into the model and how the loss is calculated
        data, target = batch
        outputs = self(data)

        # Calculate the loss
        loss = self.loss(outputs, target)

        # Count number of correct digits
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target).sum().item()

        batch_size = outputs.shape[0]

        # Log loss and number of correct predictions
        self.log("training_loss", loss, on_epoch=True, on_step=False)
        self.log(
            "training_accuracy", correct / batch_size, on_epoch=True, on_step=False
        )

        # training_step returns the loss
        return loss

    def test_step(self, batch):
        # test_step-function specifies how data is fed into the model and how the loss is calculated
        data, target = batch
        outputs = self(data)

        # Calculate the loss
        loss = self.loss(outputs, target)

        # Count number of correct digits
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target).sum().item()

        batch_size = outputs.shape[0]

        # Log loss and number of correct predictions
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        self.log("test_accuracy", correct / batch_size, on_epoch=True, on_step=False)

        # training_step returns the loss
        return loss

    def configure_optimizers(self):
        # configure_optimizers-function specifies how the optimizer is created
        return torch.optim.AdamW(self.layers.parameters())
