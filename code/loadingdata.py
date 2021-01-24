from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import torch.nn as nn

from qmath import calc_vo_relative, calc_vo_relative_logq

from vlocnet import SiameseNetwork as SiamNN

from customdataset import scene, Rescale, RandomCrop, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fire_trainset = scene(
					scene_data='fire/',
					train=True,
					transform=transforms.Compose([
						Rescale(256),
						RandomCrop(224),
						ToTensor()
                   ]))


trainloader = DataLoader(fire_trainset, batch_size=2, num_workers=2)
import math
c=0

siamnn = SiamNN().double().cuda()

z = len(trainloader)

loss = nn.MSELoss()
for param in siamnn.parameters():
	param.requires_grad = True
optimizer = optim.Adam(siamnn.parameters(), lr=1e-4)
running_loss = 0.0
mo = nn.Sequential(*list(siamnn.children())[0]).cuda()
print(mo)
exit(0)
print("starting training...")
for epoch in range(120):
	c=0
	for i in trainloader: # goes 2 images and 2 positions at a time, through the dataset.
		img_minus1 = i['image'][0].double().numpy()
		img_1 = i['image'][1].double().numpy()
		# output = siamnn.forward(x,xminus1)
		xminus1 = torch.from_numpy(np.array([img_minus1])).double()
		# p_minus1 = np.array([i['pose'][0].double().numpy()[:-1]])
		p_minus1 = np.array([i['pose'][0].double().numpy()])

		x = torch.from_numpy(np.array([img_1])).double()
		# p = np.array([i['pose'][1].double().numpy()[:-1]])
		p = np.array([i['pose'][1].double().numpy()])
		# torch.from_numpy()
		# print(calc_vo_relative_logq(torch.from_numpy(p_minus1),torch.from_numpy(p)))
		xminus1 = xminus1.to(device)
		x = x.to(device)
		output = siamnn(x, xminus1)
		l2loss = torch.sqrt(loss(output,calc_vo_relative(torch.from_numpy(p_minus1),torch.from_numpy(p)).to(device)))
		running_loss += l2loss.item()
		if (c+1)%32 == 0:
			print('[%d, %5d] loss: %.3f' %(epoch + 1, c + 1, running_loss / 16))
			optimizer.zero_grad()
			l2loss.backward()
			optimizer.step()
			running_loss = 0.0
		c+=1

	if epoch % 10 == 0:
		PATH = "models/epoch"+str(epoch+1)+".pth"
		torch.save(siamnn, PATH)
		# model = torch.load(PATH)
		# model.eval()

'''
n1 = calc_vo_relative(torch.from_numpy(p_minus1),torch.from_numpy(p)).numpy().flatten()
n2 = output.cpu().detach().numpy()
print(n1.shape,n2.shape)
eucdist = np.linalg.norm(n1-n2)
mloss = math.sqrt(np.sum((n1-n2)**2))
print(mloss)
print(l2loss)
print(eucdist)
exit(0)
'''