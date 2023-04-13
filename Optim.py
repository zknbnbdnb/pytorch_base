from torch.nn import L1Loss
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.optim import SGD


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, 
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=1)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2), 
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.model1(x)
        return x
    
NerveNetwork = NN()

loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(NerveNetwork.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = NerveNetwork(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss += result_loss
    print(running_loss)