# -*- coding:utf-8 -*-
import os
import json
import random
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

img_size = 64

# img = Image.open(img_path)
# print(img.size)
# preprocess = transforms.Compose([transforms.Resize(img_size),
#                                  transforms.CenterCrop(img_size),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                       std=[0.229, 0.224, 0.225])])
# img_ = preprocess(img)
#


# img_.shape
# img_ = cv2.cvtColor(img_.permute(1,2,0).numpy(), cv2.COLOR_RGB2BGR)
# cv2.imshow('img',img_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# class Age_Loss(torch.nn.Module):
#     def __init__(self):
#         super(Age_Loss, self).__init__()
#
#     def forward(self, pred, target):
#         l2_loss = abs(pred - target)
#
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
#
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)


# def denormalizing(tensor):
# 	for t, m, s in zip(tensor, mean, std):
# 		t.mul_(s).add_(m)
# 	# The normalize code -> t.sub_(m).div_(s)
# 	return tensor

def denormalizing(tensor, mean, std):
	for t, m, s in zip(tensor, mean, std):
		t.mul_(s).add_(m)
	# t.sub_(m).div_(s)
	return tensor


def image_preprocess(img_path=None):
	preprocess = transforms.Compose([transforms.Resize(img_size + 4),
	                                 transforms.RandomCrop(img_size),
	                                 transforms.RandomHorizontalFlip(),
	                                 transforms.ToTensor(),
	                                 # transforms.Normalize(mean=mean,
	                                 #                      std=std)
	                                 ])
	if img_path is None:
		return preprocess
	else:
		img = Image.open(img_path)
		img = preprocess(img)
		return img


def _age_categorization(age):
	if age < 20:
		age_ = 0
	elif 20 <= age < 30:
		age_ = 1
	elif 30 <= age < 40:
		age_ = 2
	elif 40 <= age < 50:
		age_ = 3
	elif 50 <= age < 60:
		age_ = 4
	# elif 60 <= age < 70:
	#     age_=5
	# elif 70 <= age:
	#     age_=6
	else:
		age_ = -1
	return int(age_)


class FaceImageDataset(Dataset):
	def __init__(self, info):
		self.info = info

	def __getitem__(self, idx):
		temp = self.info[idx]
		img_path = temp['image']
		assert img_path is not None, f'image path can not be None: {temp}'
		img = image_preprocess(img_path)
		gender = torch.tensor(([0, 1]), dtype=torch.float32) if temp['gender'] == 'male' else torch.tensor(([1, 0]), dtype=torch.float32)
		age = torch.tensor(_age_categorization(temp['age']), dtype=torch.float32)
		return img, gender, age

	def __len__(self):
		return len(self.info)


def split_dataset(path: list, split_rate: list):
	### load tarball dataset
	# path = ['dataset/metadata.json','dataset/train_image_unbalanced/metadata.json']
	if split_rate is None:
		split_rate = [0.8, 0.2]
	with open(path[0], 'r') as tarbal:
		tarball_df = json.load(tarbal)

	### load aihub dataset
	with open(path[1], 'r') as aihub:
		aihub_df = json.load(aihub)

	unbalanced_dataset = tarball_df + aihub_df
	random.shuffle(unbalanced_dataset)
	dataset = []
	age_max_count = 32000
	gender_max_count = 640000
	male_count = 0
	female_count = 0
	count_10 = 0
	count_20 = 0
	count_30 = 0
	count_40 = 0
	count_50 = 0
	count_60 = 0
	count_70 = 0

	for dict_ in unbalanced_dataset:
		if dict_['gender'] == 'female':
			if female_count <= male_count and female_count <= gender_max_count:
				female_count += 1
				if dict_['age'] < 20:
					if count_10 < age_max_count:
						dataset.append(dict_)
						count_10 += 1
				elif 20 <= dict_['age'] < 30:
					if count_20 < age_max_count:
						dataset.append(dict_)
						count_20 += 1
				elif 30 <= dict_['age'] < 40:
					if count_30 < age_max_count:
						dataset.append(dict_)
						count_30 += 1
				elif 40 <= dict_['age'] < 50:
					if count_40 < age_max_count:
						dataset.append(dict_)
						count_40 += 1
				elif 50 <= dict_['age'] < 60:
					if count_50 < age_max_count:
						dataset.append(dict_)
						count_50 += 1
				else:
					continue
			else:
				continue
		else:
			male_count += 1
			if male_count <= gender_max_count:
				if dict_['age'] < 20:
					if count_10 < age_max_count:
						dataset.append(dict_)
						count_10 += 1
				elif 20 <= dict_['age'] < 30:
					if count_20 < age_max_count:
						dataset.append(dict_)
						count_20 += 1
				elif 30 <= dict_['age'] < 40:
					if count_30 < age_max_count:
						dataset.append(dict_)
						count_30 += 1
				elif 40 <= dict_['age'] < 50:
					if count_40 < age_max_count:
						dataset.append(dict_)
						count_40 += 1
				elif 50 <= dict_['age'] < 60:
					if count_50 < age_max_count:
						dataset.append(dict_)
						count_50 += 1
				# elif dict_['age'] == 60:
				#     if count_60 < max_count:
				#         dataset.append(dict_)
				#         count_60 += 1
				#     else:
				#         continue
				# elif dict_['age'] > 60:
				#     if count_70 < max_count:
				#         dataset.append(dict_)
				#         count_70 += 1
				#     else:
				#         continue
				else:
					continue
			else:
				continue
	# print(f'count_10: {count_10}')
	# print(f'count_20: {count_20}')
	# print(f'count_30: {count_30}')
	# print(f'count_40: {count_40}')
	print(f'malecount: {male_count}')
	print(f'femalecount: {female_count}')
	random.shuffle(dataset)
	size = len(dataset)
	train_set = dataset[0: int(size * split_rate[0])]
	validation_set = dataset[int(size * (split_rate[0])):]
	return train_set, validation_set


#
#
#
def save_tensor_as_image(tensor, path):
	from cv2 import cv2
	### tensor should be (c,h,w) format
	##f"val_images/{time.time()}_{int(torch.argmax(gender, dim=1))}_{int(age)}.png"
	assert path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'), 'pass correct image file format to the save path'
	tensor = denormalizing(tensor, mean=[0, 0, 0], std=[255, 255, 255])
	img_ = cv2.cvtColor(tensor.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
	cv2.imwrite(path, img_)
#
# from torch.utils.data import DataLoader
#
# img_path = [r'dataset/train_image_unbalanced/metadata.json', r'dataset/train_image_unbalanced/metadata.json']
#
# train, val = split_dataset(img_path, split_rate=[0.1, 0.1])
# from cv2 import cv2
# import time, numpy as np
#
# dataset = FaceImageDataset(train)
# val_data_loader = DataLoader(dataset=dataset, batch_size=1)
# count = 0
# for img, gender, age in val_data_loader:
# 	# print(img)
# 	img = img.squeeze(0)
# 	img = denormalizing(img,mean=[0,0,0],std=[255,255,255])
# 	#
# 	# print(img)
# 	# break
# 	count += 1
# 	if count > 5: break
# 	img_ = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
# 	cv2.imwrite(f"val_images/{time.time()}_{int(torch.argmax(gender, dim=1))}_{int(age)}.png", img_)
# cv2.imshow('img',img_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# img_path=r'G:\locs_projects\on_working\Face_detection\val_images\10579A36-with-mask.jpg'
# img = Image.open(img_path)
# preprocess = transforms.Compose([transforms.Resize(img_size + 4),
#                                  transforms.RandomCrop(img_size),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  # transforms.Normalize(mean=mean,
#                                  #                      std=std)
#                                  ])
# print(np.array(img))
# # img = np.array(img)
# img_ = preprocess(img)
# torch.min(img_)
# torch.max(img_)
# print(img_)
# img = denormalizing_(img_,mean=[0,0,0],std=[255,255,255])
# torch.min(img)
# torch.max(img)

#
#

#
# def denormalizing__(tensor):
# 	inv_normalize = transforms.Compose([transforms.ToTensor(),
# 	                                    transforms.Normalize(mean=[0,0,0],std=[255,255,255])
# 	                                    ])
# 	return inv_normalize(tensor)
