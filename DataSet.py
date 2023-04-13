from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir # 图片上上级目录（train）
        self.label_dir = label_dir # 图片上级目录 Eg：（ants_images）
        self.path = os.path.join(root_dir, label_dir) # 拼接地址
        self.img_path = os.listdir(self.path) # 将图片存储成列表
        
    def __getitem__(self, idx):
        img_name = self.img_path[idx] # 图片第几个的索引
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) # 将所有路径拼接
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
        
    def __len__(self):
        return len(self.img_path) # 数据集长度

root_dir = 'Data_ji/train' 

ants_label_dir = 'ants_image'

ants_dataset = MyData(root_dir, ants_label_dir) # 实例化

# img, label = ants_dataset[0]
# img.show() 图片展示