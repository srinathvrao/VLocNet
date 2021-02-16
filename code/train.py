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

from models import SiameseNetwork as SiamNN, GlobalPoseNetwork as GPoseNN

from customdataset import scene, Rescale, RandomCrop, ToTensor
print("finished imports.")

fire_trainset = scene(
					scene_data='heads/',
					train=True,
					transform=transforms.Compose([
						Rescale(256),
						RandomCrop(224),
						ToTensor()
                   ]))

fire_val_test_set = scene(
					scene_data='heads/',
					train=False,
					transform=transforms.Compose([
						Rescale(256),
						RandomCrop(224),
						ToTensor()
                   ]))
print("Dataset declared.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device selected:",device)
batch_size = 20
batch_size2 = 250
trainloader = DataLoader(fire_trainset, batch_size=batch_size, num_workers=2,shuffle=False)
trainloader2 = DataLoader(fire_trainset, batch_size=batch_size2, num_workers=2,shuffle=False)
valtestloader = DataLoader(fire_val_test_set, batch_size=batch_size2, num_workers=2,shuffle=False)
print("Dataloaders declared.")

c=0
vlocn = VLocNet().double().cuda()
print("Vlocnet initialized.")
z = len(trainloader)
v = len(valtestloader) // 2 # <- 2 is number of test splits given. using 1st split for validation.
mseloss = nn.MSELoss()
for param in vlocn.parameters():
	param.requires_grad = True
optimizer = optim.Adam(vlocn.parameters(), lr=1e-4)
l = len(trainloader)*batch_size
l2 = len(valtestloader)*batch_size2 // 2

print("Starting Training")
for epoch in range(120):
	c=0
	for k, train_batch in enumerate(trainloader,0):
		train_batch['image'] = train_batch['image'].to(device)
		y_true = torch.empty(0,7).cuda()
		pose_true = torch.empty(0,7).cuda()
		y_pred, pose_pred = vlocn(train_batch['image'],batch_size) # predicting output for both tasks
		torch.cuda.empty_cache()
		poses = train_batch['pose'].unsqueeze(1)
		del train_batch
		torch.cuda.empty_cache()
		for i in range(0,batch_size-1): # calculating ground truth for task 1. (task 2 gt is already there)
			p_minus1 = poses[i]
			p_1 = poses[i+1]
			actual_output = calc_vo_relative(p_minus1,p_1).cuda()
			y_true = torch.cat((y_true, actual_output), 0)
			pose_true = torch.cat((pose_true, p_1.cuda()), 0)
			del p_minus1, p_1, actual_output
			torch.cuda.empty_cache()
		
		loss = mseloss(y_pred, y_true) + mseloss(pose_pred,pose_true)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		del y_pred, y_true, pose_pred, pose_true, loss
		torch.cuda.empty_cache()

		c+=batch_size
		print(str(c)+"/"+str(l),end=' ')
		if c%100==0:
			z=0
			st = "Epoch "+str(epoch+1)+" "+str(c)+"/"+str(l)+" ... "
			print("\n"+st, end='')	
			with torch.no_grad():
			  y_pred = torch.empty(0,7).cuda()
			  y_true = torch.empty(0,7).cuda()
			  pose_pred = torch.empty(0,7).cuda()
			  pose_true = torch.empty(0,7).cuda()
			  for j, t_batch in enumerate(trainloader2,0):
			    t_batch['image'] = t_batch['image'].to(device)
			    poses = t_batch['pose'].unsqueeze(1)
			    y_p, pose_p = vlocn(t_batch['image'],batch_size2) # predicting output for both tasks
			    y_pred = torch.cat((y_pred, y_p), 0)
			    pose_pred = torch.cat((pose_pred, pose_p), 0)
			    del y_p, pose_p, t_batch
			    for i in range(0,batch_size2-1): # calculating ground truth for task 1. (task 2 gt is already there)
			      p_minus1 = poses[i]
			      p_1 = poses[i+1]
			      actual_output = calc_vo_relative(p_minus1,p_1).cuda()
			      y_true = torch.cat((y_true, actual_output), 0)
			      pose_true = torch.cat((pose_true, p_1.cuda()), 0)
			      del p_minus1, p_1, actual_output
			      torch.cuda.empty_cache()
			  train_loss = mseloss(y_pred, y_true) + mseloss(pose_pred,pose_true)
			  print("trainset loss: ",train_loss.item(), end=' ')
			  del train_loss, y_pred, y_true, pose_pred, pose_true
			  torch.cuda.empty_cache()

			  zz = 0
			  y_pred = torch.empty(0,7).cuda()
			  y_true = torch.empty(0,7).cuda()
			  pose_pred = torch.empty(0,7).cuda()
			  pose_true = torch.empty(0,7).cuda()
			  for j, t_batch in enumerate(valtestloader,0):
			    t_batch['image'] = t_batch['image'].to(device)
			    poses = t_batch['pose'].unsqueeze(1)
			    y_p, pose_p = vlocn(t_batch['image'],batch_size2) # predicting output for both tasks
			    y_pred = torch.cat((y_pred, y_p), 0)
			    pose_pred = torch.cat((pose_pred, pose_p), 0)
			    del y_p, pose_p, t_batch
			    for i in range(0,batch_size2-1): # calculating ground truth for task 1. (task 2 gt is already there)
			      p_minus1 = poses[i]
			      p_1 = poses[i+1]
			      actual_output = calc_vo_relative(p_minus1,p_1).cuda()
			      y_true = torch.cat((y_true, actual_output), 0)
			      pose_true = torch.cat((pose_true, p_1.cuda()), 0)
			      del p_minus1, p_1, actual_output
			      torch.cuda.empty_cache()
			    zz+=batch_size2
			    if zz==l2:
			      break
			  print(y_pred.shape, y_true.shape, pose_pred.shape, pose_true.shape)
			  val_loss = mseloss(y_pred, y_true) + mseloss(pose_pred,pose_true)
			  print("valset loss: ",val_loss.item(), end=' ')
			  del val_loss, y_pred, y_true, pose_pred, pose_true
			  torch.cuda.empty_cache()
     
	if (epoch+1) % 3 == 0:
		print("Saving Checkpoint..")
		PATH = "models/epoch"+str(epoch+1)+".pth"
		torch.save(vlocn, PATH)

		# model = torch.load(PATH)
		# model.eval()
