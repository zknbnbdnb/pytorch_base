from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs') # 创建日记文件夹

image_path = 'Data_ji/train/ants_image/0013035.jpg' # 相对路径

img_PIL = Image.open(image_path) # 打开图片地址

img_array = np.array(img_PIL) # 将图片转化为numpy向量的格式

writer.add_image('test', img_array, 1, dataformats='HWC') 
# 添加图片 add_image对shape有要求，numpy的话通道在后，需要定义shape



'''for i in range(100):
    writer.add_scalar('y=x', i, i) # 绘制y=x的图像，第一个i是y轴，第二个是x轴
    # 添加标量 '''

writer.close() # 关闭