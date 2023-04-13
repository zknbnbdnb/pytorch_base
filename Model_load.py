import torch
import torchvision
from torch import nn

model = torch.load('vgg16_method1.pth')
print (model)

vgg16 = torchvision.models.vgg16(pretrained = False)
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print (vgg16)

# model2 = torch.load('test.pth')
# print (model2)