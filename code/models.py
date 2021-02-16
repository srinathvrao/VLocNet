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

class VLocNet(nn.Module):
	def __init__(self):
		super(VLocNet, self).__init__()
		# Setting up the Sequential of CNN Layers
		res50 = models.resnet50(pretrained=True)
		res50 = res50.double()
		self.sharednn = nn.Sequential(*list(res50.children())[:-4]).cuda()
		self.siam_mid = nn.Sequential(*list(res50.children())[-4]).cuda()
		self.siam_fc = nn.Sequential(*list(res50.children())[-3])
		self.siam_fc.add_module("fc",nn.Sequential(
			nn.AvgPool2d(kernel_size=(5,5)),
			nn.Dropout2d(p=0.2),
			nn.Flatten(0),
			nn.Linear(4096, 1024),
			nn.ReLU(inplace=True),
			# Final Dense Layer
			nn.Linear(1024, 7)
			).cuda()
		)

		self.gpose = nn.Sequential(*list(res50.children())[-4:-2]).cuda()
		self.gpose.add_module("fc",nn.Sequential(
			nn.AvgPool2d(kernel_size=(5,5)),
			nn.Dropout2d(p=0.2),
			nn.Flatten(0),
			nn.Linear(2048, 1024),
			nn.ReLU(inplace=True),
			# Final Dense Layer
			nn.Linear(1024, 7)
			).cuda()
		)

		for param in self.sharednn.parameters():
			param.requires_grad = True
		for param in self.siam_mid.parameters():
			param.requires_grad = True
		for param in self.siam_fc.parameters():
			param.requires_grad = True
		for param in self.gpose.parameters():
			param.requires_grad = True
	
	def forward(self, images, batch_size):

		y_pred = torch.empty(0,7).cuda()
		pose_pred = torch.empty(0,7).cuda()
		images = images.unsqueeze(1)

		for i in range(0,batch_size-1):
			img_minus1 = images[i]
			img_1 = images[i+1]
			x_sharedmap = self.sharednn(img_1)
			xminus1_sharedmap = self.sharednn(img_minus1)
			cat_fmap = torch.cat((self.siam_mid(xminus1_sharedmap), self.siam_mid(x_sharedmap)), 0)
			task1_op = self.siam_fc(cat_fmap).unsqueeze(0)
			task2_op = self.gpose(x_sharedmap).unsqueeze(0)
			y_pred = torch.cat((y_pred, task1_op), 0)
			pose_pred = torch.cat((pose_pred, task2_op), 0)

			del img_minus1, img_1, x_sharedmap, xminus1_sharedmap, cat_fmap, task1_op, task2_op
			torch.cuda.empty_cache()
