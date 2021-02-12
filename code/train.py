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
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device selected.")
fire_trainset = scene(
					scene_data='fire/',
					train=True,
					transform=transforms.Compose([
						Rescale(256),
						RandomCrop(224),
						ToTensor()
                   ]))
print("trainset declared.")
batch_size = 20
trainloader = DataLoader(fire_trainset, batch_size=batch_size, num_workers=2)
import math
c=0
print("trainset loaded.")
siamnn = SiamNN().double().cuda()
gposenn = GPoseNN(siamnn).double().cuda()
print("declared model variables.")
z = len(trainloader)

mseloss = nn.MSELoss()
for param in siamnn.parameters():
	param.requires_grad = True
for param in gposenn.parameters():
	param.requires_grad = True
l = len(trainloader)*batch_size
print("Starting Training")
for epoch in range(120):
	c=0
	printProgressBar(0, l, prefix = 'Epoch '+str(epoch+1)+':', suffix = 'Complete', length = 50)
	for batch in trainloader: # loads batch_size number of images and poses
		y_pred = torch.empty(0,7).cuda()
		y_true = torch.empty(0,7).cuda()

		pose_pred = torch.empty(0,7).cuda()
		pose_true = torch.empty(0,7).cuda()

		optimizer = optim.Adam(siamnn.parameters(), lr=1e-4)

		for i in range(0,batch_size-1): # considering 2 samples (i, i+1) at a time in a batch
			img_minus1 = batch['image'][i].unsqueeze(0).to(device)
			img_1 = batch['image'][i+1].unsqueeze(0).to(device)
			predicted_output = siamnn(img_1,img_minus1).unsqueeze(0)
			y_pred = torch.cat((y_pred, predicted_output), 0)
			
			p_minus1 = batch['pose'][i].unsqueeze(0)
			p_1 = batch['pose'][i+1].unsqueeze(0)
			actual_output = calc_vo_relative(p_minus1,p_1).cuda()
			y_true = torch.cat((y_true, actual_output), 0)
			
			del img_minus1, img_1, predicted_output, p_minus1, p_1, actual_output

		loss = mseloss(y_pred, y_true)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		gposenn.update_siam(siamnn)

		del y_pred, y_true

		y_pred = torch.empty(0,7).cuda()
		y_true = torch.empty(0,7).cuda()

		gposeoptim = optim.Adam(gposenn.parameters(), lr=1e-4)

		for i in range(0,batch_size):
			img_1 = batch['image'][i].unsqueeze(0).to(device)
			p_1 = batch['pose'][i].unsqueeze(0).to(device)
			pred_gpose = gposenn(img_1).unsqueeze(0)
			y_pred = torch.cat((y_pred, pred_gpose), 0)
			y_true = torch.cat((y_true, p_1), 0)

			del img_1, p_1, pred_gpose

		gposeloss = mseloss(y_pred, y_true)
		gposeoptim.zero_grad()
		gposeloss.backward()
		gposeoptim.step()
		siamnn.update_siam(gposenn)

		del y_pred, y_true
		c+=batch_size
		
		printProgressBar(c, l, prefix = 'Epoch '+str(epoch+1)+':', suffix = 'vo_loss: '+'{:.4f}'.format(loss)+' gpose_loss: '+'{:.4f}'.format(gposeloss)  , length = 50)
		
	if epoch % 5 == 0:
		print("Saving Checkpoint..")
		PATH = "models/epoch"+str(epoch+1)+".pth"
		torch.save(siamnn, PATH)
	
		# model = torch.load(PATH)
		# model.eval()