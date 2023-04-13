import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(
    "./dataset", train=False, transform=torchvision.transforms.ToTensor())
# torchvision提供数据集在datase文件夹中，用transform进行张量处理
test_loader = DataLoader(dataset=test_data, batch_size=64,
                         shuffle=True, num_workers=0, drop_last=False)
# DataLoader加载数据，batch_size用于一次性加载几次数据，shuffle是表示是否打乱加载
# num_worker为0不会报错，drop_last是表示是否丢弃余数，例如100/3剩下一个数据是否舍去

writer = SummaryWriter("logs")

step = 0
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
    writer.add_images("test_data", imgs, step)
    step += 1

writer.close()
