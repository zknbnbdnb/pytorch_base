
from torch.utils.tensorboard import SummaryWriter
from models import *
from torch import nn
from torch.utils.data import DataLoader
import torchvision


# 准备数据
train_data = torchvision.datasets.CIFAR10(root='D:\pytorch\pytorch基础\dataset',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root='D:\pytorch\pytorch基础\dataset',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为：{}".format(train_data_size))
print("测试数据集长度为：{}".format(test_data_size))

# 加载数据
train_dataloader = DataLoader(train_data, batch_size=200)
test_dataloader = DataLoader(test_data, batch_size=200)


# 准备模型
cnn = NN()
cnn = cnn.cuda()  # 模型创建可以用cuda

# 设置损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.cuda()  # 损失函数能用cuda

# 设置优化器
learning_rate = 1e-2

optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

# 正式训练
total_train_step = 0
total_test_step = 0
epoch = 200

writer = SummaryWriter("./logs/")

for i in range(epoch):
    print("----第{}轮训练开始----".format(i+1))
    cnn.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()  # 输入和标注能用cuda
        outputs = cnn(imgs)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train'_loss", loss.item(), total_train_step)

    cnn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = cnn(imgs)
            loss = loss_func(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step += 1

    torch.save(cnn, "CNN{}.pth".format(i+1))
    print("模型已保存")

writer.close()
