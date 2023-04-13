from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img = Image.open('Data_ji/train/ants_image/0013035.jpg')

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image('ToTensor', img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_normal = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_normal = trans_normal(img_tensor)
print(img_normal[0][0][0]) # 2*0.3137-1 = -0.3725
writer.add_image('Normalize', img_normal, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((256, 256))
img_resize = trans_resize(img) # PIL
img_resize = trans_totensor(img_resize) # PIL -> Tensor
writer.add_image('Resize', img_resize, 0)
print(img_resize)

# Compose - Resize-2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# PIL -> PIL -> Tensor
img_resize_2 = trans_compose(img)
writer.add_image('Resize', img_resize_2, 1)

writer.close()