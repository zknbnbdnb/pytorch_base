from torch.nn import L1Loss
import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential


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

for data in dataloader:
    imgs, targets = data
    outputs = NerveNetwork(imgs)
    result_loss = loss(outputs, targets)
    print(result_loss)

'''
input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss() # reduction = default / sum
result = loss(input, target)

loss_mse = nn.MSELoss() #平方差
result_1 = loss_mse(input, target)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss() # 交叉熵
result_2 = loss_cross(x, y) 

print(result)
print(result_1)
print(result_2)
'''

