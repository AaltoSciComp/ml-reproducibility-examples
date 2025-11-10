# Import needed modules
import torch

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from datamodules import MNISTDataModule
from models import SimpleMLP

import os.path


def plot_training(metrics):

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(metrics["epoch"], metrics["training_loss"])
    ax1.set_title("Loss")
    ax2.plot(metrics["epoch"], metrics["training_accuracy"])
    ax2.set_title("Accuracy")

    return fig, (ax1, ax2)


def main():

    # Set model parameters
    data_dir = "../../../data"
    batch_size = 32
    hidden_size = 20

    training_plot = "training_results.png"
    log_dir = "logs"
    experiment_name = "mnist_example"

    # Create model and datamodule
    model = SimpleMLP(hidden_size=hidden_size)
    datamodule = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)

    # Specify logger

    logger = CSVLogger(log_dir, name=experiment_name)

    # Train the model
    trainer = L.Trainer(
        max_epochs=5,
        callbacks=[TQDMProgressBar(refresh_rate=100)],
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)

    # Visualize training
    metrics = pd.read_csv(
        os.path.join(log_dir, experiment_name, "version_0", "metrics.csv")
    )
    fig, axes = plot_training(metrics)
    fig.savefig(training_plot)

    # Test the model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
