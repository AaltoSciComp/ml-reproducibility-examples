# Redo imports
import torch

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from models import SimpleMLP


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    # Set up data loaders based on datasets

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_model(model_structure, optimizer_class, device="cpu"):
    # Define function for getting model and optimizer

    model = model_structure().to(device)
    optimizer = optimizer_class(model.parameters())
    return model, optimizer


def train(
    model, dataloader, criterion, optimizer, batch_size=32, epochs=5, device="cpu"
):
    # Define training and validation functions

    model.train()

    num_batches = len(dataloader)
    num_items = len(dataloader.dataset)
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        for data, target in tqdm(dataloader, total=num_batches):
            # Copy data and targets to GPU
            data = data.to(device)
            target = target.to(device)

            # Do a forward pass
            outputs = model(data)

            # Calculate the loss
            loss = criterion(outputs, target)
            total_loss += loss.item()

            # Count number of correct digits
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == target).sum().item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss / num_batches
        accuracy = total_correct / num_items
        print(f"Average loss: {train_loss:7f}, accuracy: {accuracy:.2%}")
        losses.append(train_loss)
        accuracies.append(accuracy)

    return losses, accuracies


def validate(model, dataloader, criterion, device="cpu"):
    # Validate the model
    model.eval()

    num_batches = len(dataloader)
    num_items = len(dataloader.dataset)

    total_correct = 0
    for data, target in tqdm(dataloader, total=num_batches):
        # Copy data and targets to GPU
        data = data.to(device)
        target = target.to(device)

        # Do a forward pass
        outputs = model(data)

        # Count number of correct digits
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == target).sum().item()

    accuracy = total_correct / num_items
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


def plot_training(losses, accuracies):

    batch_index = np.arange(1, len(losses) + 1)

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(batch_index, np.asarray(losses))
    ax1.set_title("Loss")
    ax2.plot(batch_index, np.asarray(accuracies))
    ax2.set_title("Accuracy")

    return fig, (ax1, ax2)


def main():
    # ------------------
    # Training variables
    # ------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "../../data"
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=ToTensor()
    )
    test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())
    batch_size = 32

    model_structure = SimpleMLP
    optimizer_class = torch.optim.AdamW
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 5

    training_plot = "training_results.png"

    # --------------------
    # Actual training code
    # --------------------

    # Set up data loaders
    train_loader, test_loader = create_dataloaders(
        train_dataset, test_dataset, batch_size=32
    )

    # Set up model and optimizer
    model, optimizer = create_model(model_structure, optimizer_class, device=device)

    # Train the model
    losses, accuracies = train(
        model,
        train_loader,
        criterion,
        optimizer,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    # Visualize training
    fig, axes = plot_training(losses, accuracies)
    fig.savefig(training_plot)

    # Validate model
    validate(model, test_loader, criterion, device=device)


if __name__ == "__main__":
    main()
