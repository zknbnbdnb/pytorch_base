# 卷积层

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("./dataset", train=False, 
                                      transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x

NerveNetwork = NN()
# print(NerveNetwork)

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = NerveNetwork(imgs)
    print(imgs.shape)
    # torch.Size([64, 3, 32, 32])
    print(output.shape)
    # torch.Size([64, 6, 30, 30])
    
    output = torch.reshape(output, (-1, 3, 30, 30))
    # 彩图最多三通道，所有要将通道reshape到3，-1能根据你后面的数据进行调整
    # torch.Size([64, 6, 30, 30]) -> torch.Size([xxx, 3, 30, 30])
    
    writer.add_images("input", imgs, step)
    writer.add_images("NerveNetwork", output, step)
    
    step += 1

writer.close()