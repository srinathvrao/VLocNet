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

from models import VLocNet

from customdataset import scene, Rescale, RandomCrop, ToTensor
from IPython.display import HTML, display
import time
print("Finished imports.")

def progress(value, c, max=100):
    return HTML("""
        <progress
            value='{value}',
            max='{max}',
            style='width: 40%'
        >
            {value}
        </progress>
        {c}
    """.format(value=value,c=c, max=max))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device selected:",device)
fire_trainset = scene(
                    scene_data='fire/',
                    train=True,
                    transform=transforms.Compose([
                        Rescale(256),
                        RandomCrop(224),
                        ToTensor()
                   ]))
print("Dataset declared.")
batch_size = 20
trainloader = DataLoader(fire_trainset, batch_size=batch_size, num_workers=2)
print("Dataloader declared.")
import math
c=0

vlocn = VLocNet().double().cuda()
print("Vlocnet initialized.")
z = len(trainloader)

mseloss = nn.MSELoss()
for param in vlocn.parameters():
    param.requires_grad = True

optimizer = optim.Adam(vlocn.parameters(), lr=1e-4)
l = len(trainloader)*batch_size
print("Starting Training")

for epoch in range(120):
    c=0
    # out = display(progress(0, l), display_id=True)
    # printProgressBar(0, l, prefix = 'Epoch '+str(epoch+1)+':', suffix = 'Complete', length = 50)
    for batch in trainloader: # loads batch_size number of images and poses
        y_pred = torch.empty(0,7).cuda()
        y_true = torch.empty(0,7).cuda()

        pose_pred = torch.empty(0,7).cuda()
        pose_true = torch.empty(0,7).cuda()

        for i in range(0,batch_size-1): # considering 2 samples (i, i+1) at a time in a batch
            img_minus1 = batch['image'][i].unsqueeze(0).to(device)
            img_1 = batch['image'][i+1].unsqueeze(0).to(device)
            p_minus1 = batch['pose'][i].unsqueeze(0)
            p_1 = batch['pose'][i+1].unsqueeze(0)

            task1_op, task2_op = vlocn(img_1,img_minus1)
            y_pred = torch.cat((y_pred, task1_op.unsqueeze(0)), 0)
            
            actual_output = calc_vo_relative(p_minus1,p_1).cuda()
            y_true = torch.cat((y_true, actual_output), 0)

            pose_pred = torch.cat((pose_pred, task2_op.unsqueeze(0)), 0)
            pose_true = torch.cat((pose_true, p_1.cuda()), 0)

            del img_minus1, img_1, p_minus1, p_1, task1_op, task2_op, actual_output

        loss = mseloss(y_pred, y_true) + mseloss(pose_pred,pose_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del y_pred, y_true, pose_pred, pose_true
    
        c+=batch_size
        st = "Epoch "+str(epoch+1)+" "+str(c)+"/"+str(l)+" Loss: "+str(loss.item())
        print(st)
        # out.update(progress(c, st, l))
        # printProgressBar(c, l, prefix = 'Epoch '+str(epoch+1)+':', suffix = 'vo_loss: '+'{:.4f}'.format(loss)+' gpose_loss: '+'{:.4f}'.format(gposeloss)  , length = 50)
        
    if (epoch+1) % 5 == 0:
        print("Saving Checkpoint..")
        PATH = "models/epoch"+str(epoch+1)+".pth"
        torch.save(vlocn, PATH)
    
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
