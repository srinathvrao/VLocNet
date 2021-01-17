from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os

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

c=0

res50 = models.resnet50(pretrained=True)

siamtminus1_res14 = nn.Sequential(*list(res50.children())[:-3])
siamtminus1_res14.globalavgpool = nn.AvgPool2d(kernel_size=(5,5))
siamtminus1_res14.flatt = nn.Flatten(1)
siamtminus1_res14.fc1 = nn.Linear(4096,1024)
siamtminus1_res14.fc2 = nn.Linear(1024,7)

for param in siamtminus1_res14.parameters():
	param.requires_grad = True

# siamtminus1_res14.dropout1 = nn.Dropout2d(0.2)

# siamtminus1_res14.globalavgpool = nn.AvgPool2d(kernel_size=(4,4))
# siamtminus1_res14.fc2 = nn.Linear(1024, 7)


siamtminus1_res14 = siamtminus1_res14.double()

z = len(trainloader)
for i in trainloader:
	if c>0 and c<z-1:
		# use z['image'] and i['image'][0] 
		pass
	# print(i['image'].shape) # 2 images
	# print(i['pose'].shape) # 2 poses
	img = i['image'][0].double().numpy()
	imgtorch = torch.from_numpy(np.array([img])).double()
	avgp = siamtminus1_res14(imgtorch)
	print(avgp)








	# saved for the last
	z = {'image':i['image'][1] , 'pose': i['pose'][1]}
	
	break