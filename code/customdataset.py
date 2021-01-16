from __future__ import print_function, division
import os

import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import utils, models
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R

import warnings
warnings.filterwarnings("ignore")

class scene(Dataset):
	"""A scene from the Microsoft 7-scenes dataset."""

	def __init__(self,scene_data,train=False,transform=None):
		"""
		Args:
			scene_data (string): Path to the folder containing the sequences of the scene.
			train (bool): Refers to test set, if set to false. Otherwise, train set.
			transform (callable, optional): Optional transform to be applied on a sample.			
		"""
		self.scene_data_path = scene_data
		self.train = train
		self.transform = transform
		self.seq_nums = []
		self.seq_image_count = []
		sp = []
		trainsp = open(os.path.join(self.scene_data_path,"TrainSplit.txt"))
		testsp = open(os.path.join(self.scene_data_path,"TestSplit.txt"))
		if self.train:
			sp = trainsp
		else:
			sp = testsp
		a=0
		for line in sp:
			seq_num = int(line[:-1].split("sequence")[1])
			seq=""
			if seq_num<10:
				seq = "0"+str(seq_num)
			else:
				seq = str(seq_num)
			self.seq_nums.append(seq)
			b = (len(os.listdir(os.path.join(self.scene_data_path,"seq-"+seq))) // 3)
			self.seq_image_count.append(b)
			a += b
		self.numsequences = a

	def __len__(self):
		return sum(self.seq_image_count)

	def __getitem__(self, idx):

		"""
		Args:
			idx (int): Refers to the frame number in the entire split
		"""
		i,c = 0,0
		seq_idx = 0
		idx2 = idx
		n_counts = len(self.seq_image_count)
		for i in range(n_counts):
			c+=self.seq_image_count[i]
			if idx<c:
				seq_idx = i
				break
			else:
				idx2 -= self.seq_image_count[i]

		imid = ""
		if idx2<10:
			imid = "00000"+str(idx2)
		elif idx2<100:
			imid = "0000"+str(idx2)
		else:
			imid = "000"+str(idx2)
		image = io.imread(os.path.join(self.scene_data_path,"seq-"+self.seq_nums[seq_idx],"frame-"+imid+".color.png")) # frame-000000.color.png
		posefile = open(os.path.join(self.scene_data_path,"seq-"+self.seq_nums[seq_idx],"frame-"+imid+".pose.txt"))
		pose = []
		for line in posefile:
			p = line[:-1].split()
			p = np.array([float(x) for x in p])
			pose.append(p)
		pose = np.array(pose)
		
		# converting 4x4 matrix to rotation and translation, 7-dimensions

		rotationmatrix = pose[:3,:3]
		translation = pose[:3,3]
		r = R.from_dcm(rotationmatrix)
		rotation = r.as_quat()
		pose = np.concatenate([translation, rotation])
		sample = {'image': image, 'pose': pose}
		
		if self.transform:
			sample = self.transform(sample)

		return sample




class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'pose': pose}

class RandomCrop(object):
	"""Crop randomly the image in a sample."""
	
	"""
	Args:
		output_size (tuple or int): Desired output size. If int, square crop is made.
	"""
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, pose = sample['image'], sample['pose']
		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h,
		              left: left + new_w]
		return {'image': image, 'pose': pose}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'pose': torch.from_numpy(pose)}
