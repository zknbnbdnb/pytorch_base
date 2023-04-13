import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.tensorboard import SummaryWriter

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        '''self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d 计算公式
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2) 
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        # self.linear1 = Linear(10240, 64) #  错误的，用于测试
        self.linear2 = Linear(64, 10)'''
        # 下面是简化版
        
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2), # ?????????
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )
        
    def forward(self, x):
        '''x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)''' 
        
        x = self.model1(x)
        return x
    
NerveNetwork = NN()
# print(NerveNetwork) #看结构
input = torch.ones((64, 3, 32, 32)) # 创建个64batch_size的3通道32*32的全1张量 用于测试神经网络准确性
output = NerveNetwork(input)
print(output.shape) 

writer = SummaryWriter("./logs")
writer.add_graph(NerveNetwork, input)
writer.close()