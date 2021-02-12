from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import time
import transforms3d.quaternions as txq

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		# Setting up the Sequential of CNN Layers
		res50 = models.resnet50(pretrained=True)
		res50 = res50.double()

		self.siamnn = nn.Sequential(*list(res50.children())[:-2])
		self.fcOut = nn.Sequential(*list(self.siamnn.children())[-1:])
		self.siamnn = nn.Sequential(*list(self.siamnn.children())[:-1]).cuda()
		self.fcOut.add_module("fc",nn.Sequential(
			nn.AvgPool2d(kernel_size=(5,5)),
			nn.Dropout2d(p=0.2),
			nn.Flatten(0),
			nn.Linear(4096, 1024),
			nn.ReLU(inplace=True),
			# Final Dense Layer
			nn.Linear(1024, 7)
			).cuda()
		)
		for param in self.siamnn.parameters():
			param.requires_grad = True
		for param in self.fcOut.parameters():
			param.requires_grad = True
	
	def update_siam(self, gposenn):
		sharednn = nn.Sequential()
		sharednn.add_module("shared",nn.Sequential(*list(gposenn.children())[0]))
		sharednn.add_module("siamn6",nn.Sequential(*list(self.siamnn.children())[-1]))
		self.siamnn = sharednn.cuda()
		for param in self.siamnn.parameters():
			param.requires_grad = True

	def forward(self, x , xminus1):
		# halfnn = nn.Sequential(*list(self.siamnn.children())[:-2]).cuda()
		x_fmap, xminus1_fmap = self.siamnn(x), self.siamnn(xminus1)
		cat_fmap = torch.cat((xminus1_fmap, x_fmap), 0)
		output = self.fcOut(cat_fmap)
		return output

class GlobalPoseNetwork(nn.Module):
	def __init__(self, siamnn):
		super(GlobalPoseNetwork, self).__init__()
		# Setting up the Sequential of CNN Layers
		res50 = models.resnet50(pretrained=True)
		res50 = res50.double()

		self.shared_nn = nn.Sequential()
		self.gpose = nn.Sequential(*list(res50.children())[-4:-2]).cuda()
		self.gpose.add_module("fc",nn.Sequential(
			nn.AvgPool2d(kernel_size=(5,5)),
			nn.Dropout2d(p=0.2),
			nn.Flatten(0),
			nn.Linear(2048, 1024),
			nn.ReLU(inplace=True),
			# Final Dense Layer
			nn.Linear(1024, 7)
			).cuda())
		for param in self.gpose.parameters():
			param.requires_grad = True
		
	def update_siam(self, siam):
		self.shared_nn = nn.Sequential(*list(siam.children())[0][:-1]).cuda()
		for param in self.shared_nn.parameters():
			param.requires_grad = True

	def forward(self, x):
		shared_fmap = self.shared_nn(x)
		gpose_op = self.gpose(shared_fmap)
		return gpose_op # <- change this