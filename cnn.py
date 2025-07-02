import torch, torchvision, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt, numpy as np

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparams
num_epochs, batch_size, lr = 5, 4, 0.001

# Transform: Convert images to tensors & normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

# Load CIFAR-10 train/test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Load data in batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Class labels
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# Function to show images
def imshow(img):
    img = img / 2 + 0.5                          # unnormalize
    npimg = img.numpy()                          # convert to numpy
    plt.imshow(np.transpose(npimg, (1,2,0)))     # change format for imshow
    plt.show()

# Show sample images from train loader
dataiter = iter(train_loader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

# CNN architecture
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)          # Conv layer 1 (input: 3, output: 6, kernel: 5)
        self.pool = nn.MaxPool2d(2, 2)           # Max pooling layer (2x2)
        self.conv2 = nn.Conv2d(6, 16, 5)         # Conv layer 2 (6 → 16)
        self.fc1 = nn.Linear(16*5*5, 120)        # FC layer 1
        self.fc2 = nn.Linear(120, 84)            # FC layer 2
        self.fc3 = nn.Linear(84, 10)             # Output layer (10 classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))     # Conv1 → ReLU → Pool
        x = self.pool(F.relu(self.conv2(x)))     # Conv2 → ReLU → Pool
        x = x.view(-1, 16*5*5)                   # Flatten
        x = F.relu(self.fc1(x))                  # FC1 → ReLU
        x = F.relu(self.fc2(x))                  # FC2 → ReLU
        return self.fc3(x)                       # Output (logits)

model = ConvNet().to(device)                     # Initialize and move model to device
criterion = nn.CrossEntropyLoss()                # Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # Optimizer

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)                    # Forward
        loss = criterion(outputs, labels)        # Compute loss

        optimizer.zero_grad()                    # Clear gradients
        loss.backward()                          # Backpropagation
        optimizer.step()                         # Update weights

        if (i+1) % 2000 == 0:                    # Print every 2000 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), './cnn.pth')
print('Model saved as cnn.pth')

# Evaluation (with no gradient calculation)
with torch.no_grad():
    n_correct, n_samples = 0, 0
    n_class_correct = [0]*10
    n_class_samples = [0]*10

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)         # Predicted class

        n_correct += (preds == labels).sum().item()
        n_samples += labels.size(0)

        # Per-class accuracy count
        for i in range(batch_size):
            label = labels[i]
            pred = preds[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    # Overall accuracy
    acc = 100.0 * n_correct / n_samples
    print(f'Total Accuracy: {acc:.2f}%')

    # Per-class accuracy
    for i in range(10):
        acc_i = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]:>5s}: {acc_i:.2f}%')
