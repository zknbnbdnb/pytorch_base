import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# torch.save(vgg16, 'vgg16_method1.pth')

# torch.save(vgg16.state_dict(), 'vgg16_method2.pth')


# class test(nn.Module):
#     def __init__(self):
#         super(test, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size = 3)

#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# Test = test()
# torch.save(Test, 'test.pth')

print(vgg16)
