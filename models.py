import torch
from torch import nn

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=1, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
    
    
# (features): Sequential(
# Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# ReLU(inplace=True)
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )
# (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
# (classifier): Sequential(
# Linear(in_features=25088, out_features=4096, bias=True)
# ReLU(inplace=True)
# Dropout(p=0.5, inplace=False)
# Linear(in_features=4096, out_features=4096, bias=True)
# ReLU(inplace=True)
# Dropout(p=0.5, inplace=False)
# Linear(in_features=4096, out_features=1000, bias=True)
# )
