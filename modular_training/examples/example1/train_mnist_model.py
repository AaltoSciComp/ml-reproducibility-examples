# Import needed modules
import torch

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torch import nn

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

# Set up data sets and data loaders

data_dir = "../../../data"

batch_size = 32

train_dataset = datasets.MNIST(
    data_dir, train=True, download=True, transform=ToTensor()
)
test_dataset = datasets.MNIST(data_dir, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Specify model

device = "cuda" if torch.cuda.is_available() else "cpu"


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 20), nn.ReLU(), nn.Linear(20, 10)
        )

    def forward(self, x):
        return self.layers(x)


model = SimpleMLP().to(device)
print(model)

# Specify loss and optimizer

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters())

# Train the model

model.train()

num_batches = len(train_loader)
num_items = len(train_loader.dataset)
losses = []
accuracies = []
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    total_correct = 0
    for data, target in tqdm(train_loader, total=num_batches):
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

batch_index = np.arange(1, len(losses) + 1)

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ax1.plot(batch_index, np.asarray(losses))
ax1.set_title("Loss")
ax2.plot(batch_index, np.asarray(accuracies))
ax2.set_title("Accuracy")

fig.savefig("training_results.png")

# Test the model
model.eval()

num_test_batches = len(test_loader)
num_test_items = len(test_loader.dataset)

test_total_correct = 0
for data, target in tqdm(test_loader, total=num_test_batches):
    # Copy data and targets to GPU
    data = data.to(device)
    target = target.to(device)

    # Do a forward pass
    outputs = model(data)

    # Count number of correct digits
    _, predicted = torch.max(outputs, 1)
    test_total_correct += (predicted == target).sum().item()

test_accuracy = test_total_correct / num_test_items
print(f"Testing accuracy: {test_accuracy:.2%}")
