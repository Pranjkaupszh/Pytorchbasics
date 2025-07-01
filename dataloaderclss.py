import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np, math
import torchvision
class WineDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(data[:, 1:])
        self.y = torch.from_numpy(data[:, [0]])
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
    def __len__(self): return len(self.x)

dataset = WineDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
for epoch in range(2):
    for i, (x, y) in enumerate(loader):
        if (i+1) % 5 == 0:
            print(f'Epoch {epoch+1}, Step {i+1}, x.shape={x.shape}, y.shape={y.shape}')
mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
mnist_loader = DataLoader(dataset=mnist, batch_size=3, shuffle=True)
x, y = next(iter(mnist_loader))
print(x.shape, y.shape)
