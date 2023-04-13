# 线性层

import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn 
from torch.nn import Linear 


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, 
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.linear = Linear(196608, 10)
        # torch.Size([1, 1, 1, 196608]) -> [1, 1, 1, 10]
        
    def forward(self, input):
        output = self.linear(input)
        return output
    
NerveNetwork = NN()



for data in dataloader:
    imgs, targets = data
    # print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = NerveNetwork(output)
    print(output.shape)
