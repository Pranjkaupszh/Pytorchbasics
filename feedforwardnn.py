import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size, hidden_size, num_classes = 784, 500, 10
num_epochs, batch_size, lr = 2, 100, 0.001

# MNIST dataset (28x28 grayscale images)
transform = transforms.ToTensor()  # Convert PIL image to Tensor
train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_set  = datasets.MNIST(root='./data', train=False, transform=transform)

# Load datasets in batches
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# Visualize sample test images
examples = iter(test_loader)
images, _ = next(examples)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap='gray')  # Show grayscale image
plt.show()

# Define a simple feedforward neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.relu = nn.ReLU()                          # Activation
        self.fc2 = nn.Linear(hidden_size, num_classes) # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)  # No softmax needed for CrossEntropyLoss

model = NeuralNet().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.reshape(-1, 28*28).to(device)  # Flatten images
        labels = labels.to(device)

        outputs = model(imgs)                  # Forward pass
        loss = criterion(outputs, labels)      # Compute loss

        optimizer.zero_grad()                  # Clear gradients
        loss.backward()                        # Backward pass
        optimizer.step()                       # Update weights

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluation loop (no gradient needed)
with torch.no_grad():
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)       # Get class with highest score
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    print(f'Accuracy on 10000 test images: {100 * correct / total:.2f} %')
