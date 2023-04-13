# 池化层

import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
'''
    16-33行固定模板
    datase先设置数据，数据张量化
    dataloader加载数据
    创建神经网络框架
    最后调用tensorboard
'''
dataset = torchvision.datasets.CIFAR10("./dataset", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.max_pool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.max_pool(input)
        return output


NerveNetwork = NN()

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = NerveNetwork(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
