import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(root = './dataset/', train = True,
                                        download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100,
                                            shuffle = True)

testset = torchvision.datasets.CIFAR10(root = './dataset/', train = False,
                                        download = True, transform = transform1)

testloader = torch.utils.data.DataLoader(testset, batch_size = 100,
                                            shuffle = True)

classes = ('飞机', '车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding= 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding = 1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding = 1)
        self.pool3 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding = 1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding = 1)
        self.pool4 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        
        self.conv11 = nn.Conv2d(256, 512, 3, padding = 1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding = 1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding = 1)
        self.pool5 = nn.MaxPool2d(2, 2, padding = 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        
        self.fc14 = nn.Linear(512*4*4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        
        x = x.view(-1, 512*4*4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = F.relu(self.fc16(x))
        
        return x
    def train_sgd(self, device):
        optimizer = optim.Adam(self.parameters(), lr = 0.0001)
        
        path = 'weights.tar'
        initepoch = 0
        
        if os.path.exists(path) is not True:
            loss = nn.CrossEntropyLoss()
        else:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            initepoch = checkpoint['epoch']
            loss = checkpoint['loss']
            
        for epoch in range(initepoch, 100):
            timestart = time.time()
            
            runing_loss = 0.0
            total = 0.0
            correct = 0.0 
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()
                runing_loss += l.item()
                
                if i % 500 == 499:
                    print('[%d,%5d] loss: %.4f' % (epoch, i, runing_loss / 500))
                    runing_loss = 0.0 
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    print('（每%d张）正确率为：%.3f %%' % (total, 100 * correct/ total))
                    total = 0
                    correct = 0 
                    torch.save({
                        'epoch':epoch,
                        'model_state_dict':net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':loss,
                    }, path)
                    
                print('这第%d轮epoch花费%.3f秒' % (epoch, time.time() - timestart))
        print('结束训练')
        
    def test(self, device):
        correct = 0.0 
        total = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()
                
        print('在10000张图片此网络的正确率：%.3f %%' % (100 *correct / total))

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu') 

net = Net()
net = net.to(device)
net.train_sgd(device)
net.test(device)

