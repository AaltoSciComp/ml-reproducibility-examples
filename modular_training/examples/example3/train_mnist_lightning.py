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

import hydra


def plot_training(metrics):

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(metrics["epoch"], metrics["training_loss"])
    ax1.set_title("Loss")
    ax2.plot(metrics["epoch"], metrics["training_accuracy"])
    ax2.set_title("Accuracy")

    return fig, (ax1, ax2)


@hydra.main(version_base="1.2", config_path=".", config_name="config")
def main(conf):

    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    training_plot = os.path.join(log_dir, f"{conf.experiment_name}.png")

    # Create model and datamodule
    model = SimpleMLP(hidden_size=conf.model.hidden_size)
    datamodule = MNISTDataModule(
        data_dir=conf.dataset.data_dir, batch_size=conf.dataset.batch_size
    )

    # Specify logger

    logger = CSVLogger(log_dir, name=conf.experiment_name)

    # Train the model
    trainer = L.Trainer(
        max_epochs=conf.trainer.max_epochs,
        callbacks=[TQDMProgressBar(refresh_rate=conf.trainer.refresh_rate)],
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)

    # Visualize training
    metrics = pd.read_csv(
        os.path.join(log_dir, conf.experiment_name, "version_0", "metrics.csv")
    )
    fig, axes = plot_training(metrics)
    fig.savefig(training_plot)

    # Test the model
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
