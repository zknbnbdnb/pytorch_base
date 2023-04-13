import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor() # 矢量转化
])
# 下载数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)

writer = SummaryWriter('logs')
for i in range(1000):
    img, target = train_set[i]
    writer.add_image("train_set", img, i)
# tensorboard展示1000张数据集图片
writer.close()