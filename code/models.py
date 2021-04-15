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
		self.siam_mid.add_module("miden",nn.Sequential(nn.AvgPool3d(2, stride=2).cuda()))
    
		self.siam_fc = nn.Sequential(*list(res50.children())[-3])
		self.siam_fc.add_module("fc",nn.Sequential(
			nn.AvgPool2d(kernel_size=(3,3)), # torch.Size([38, 2048, 1, 1])
			nn.Dropout2d(p=0.2),
      nn.Flatten(1)
			).cuda()
		)

		self.siamfc_end = nn.Sequential(
      nn.Linear(2048, 1024),
			nn.ReLU(inplace=True),
			# Final Dense Layer
			nn.Linear(1024, 7)
    ).cuda()

		self.gpose6layer = nn.Sequential(*list(res50.children())[-4]).cuda()
		# print(self.gpose)
		self.gpose3layer = nn.Sequential(*list(res50.children())[-3]).cuda()
  
		self.gposefc4 = nn.Sequential(
      nn.Linear(7, 200704)
    )

		self.gposefc_end = nn.Sequential(
      nn.AvgPool2d(kernel_size=(5,5)), #torch.Size([19, 2048, 1, 1])
			nn.Dropout2d(p=0.2),
			nn.Flatten(1),
      nn.Linear(4096, 2048),
			nn.ReLU(inplace=True),
      nn.Linear(2048, 1024),
			nn.ReLU(inplace=True),
			nn.Linear(1024, 7)
		).cuda()

		for param in self.sharednn.parameters():
			param.requires_grad = True
		for param in self.siam_mid.parameters():
			param.requires_grad = True
		for param in self.siam_fc.parameters():
			param.requires_grad = True
		for param in self.gpose6layer.parameters():
			param.requires_grad = True
		for param in self.gpose3layer.parameters():
			param.requires_grad = True
		for param in self.gposefc_end.parameters():
			param.requires_grad = True

	def forward(self, train_batch, batch_size, exp):
		images = train_batch['image']
		xminus1poses = train_batch['pose'].unsqueeze(1)[:-1]
		y_pred = torch.empty(0,7).cuda()
		pose_pred = torch.empty(0,7).cuda()
		xminus1_sharedmaps = self.sharednn(images[:-1])
		x_sharedmaps = self.sharednn(images[1:])
		cat_fmap = torch.cat((self.siam_mid(xminus1_sharedmaps), self.siam_mid(x_sharedmaps)), 1)
		siam_midend = self.siam_fc(cat_fmap)
		for img_feat in siam_midend:
			y_pred = torch.cat((y_pred, self.siamfc_end(img_feat).unsqueeze(0) ), 0)
		if exp in ['val','test']:
			x1 = self.gpose6layer((torch.cat((xminus1_sharedmaps, x_sharedmaps[-1].unsqueeze(0)), 0))).unsqueeze(1)
			gposefc4reshaped = self.gposefc4(xminus1poses[0]).view(x1[1].shape[0],x1[1].shape[1],x1[1].shape[2],x1[1].shape[3]) # 19, 1024, 14, 14 and 19,1,196
			cattt = torch.cat((x1[1], gposefc4reshaped), 2)
			pose_pred = torch.cat((pose_pred,    self.gposefc_end(self.gpose3layer(cattt))     ),0)
			for i in range(2,batch_size):
			    gposefc4reshaped = self.gposefc4(pose_pred[i-2]).view(x1[1].shape[0],x1[1].shape[1],x1[1].shape[2],x1[1].shape[3]) # 19, 1024, 14, 14 and 19,1,196
			    cattt = torch.cat((x1[i], gposefc4reshaped), 2)
			    pose_pred = torch.cat((pose_pred,    self.gposefc_end(self.gpose3layer(cattt))     ),0)
			return y_pred, pose_pred

		gpose_6layerop = self.gpose6layer(x_sharedmaps)
		gposefc4reshaped = self.gposefc4(xminus1poses).view(gpose_6layerop.shape[0],gpose_6layerop.shape[1],gpose_6layerop.shape[2],gpose_6layerop.shape[3]) # 19, 1024, 14, 14 and 19,1,196
		cattt = torch.cat((gpose_6layerop, gposefc4reshaped), 2)
		return y_pred, self.gposefc_end(self.gpose3layer(cattt))
