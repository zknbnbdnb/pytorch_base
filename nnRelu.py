# 非线性激活

import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, 
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataload = DataLoader(dataset, batch_size=64)

# print(output.shape)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        
    def forward(self, input):
        output = self.sigmoid(input)
        return output
    
NerveNetwork = NN()

writer = SummaryWriter("./logs")
step = 0

for data in dataload:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = NerveNetwork(imgs)
    writer.add_images("output", output, step)
    step += 1
    
writer.close()