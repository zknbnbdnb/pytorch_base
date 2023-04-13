from torch import nn
import torch

class Hello_World_NN(nn.Module):
    def __init__(self):
        super(Hello_World_NN, self).__init__()
        
        
        
    def forward(self, input):
        output = input + 1
        return output
    
    
hello_world_NN = Hello_World_NN()
x = torch.tensor(1.0)
output = hello_world_NN(x)
print(output)