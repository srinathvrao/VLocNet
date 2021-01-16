from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

from customdataset import scene, Rescale, RandomCrop, ToTensor

fire_trainset = scene(
					scene_data='fire/',
					train=True,
					transform=transforms.Compose([
						Rescale(256),
						RandomCrop(224),
						ToTensor()
                   ]))


trainloader = DataLoader(fire_trainset, batch_size=2, num_workers=2)

for i in trainloader:
	print(i['image'].shape) # 2 images
	print(i['pose']) # 2 poses
	
	break