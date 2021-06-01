# -*- coding:utf-8 -*-
import os
import random

import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from train_module.data_helper import FaceImageDataset, split_dataset
from torch.utils.tensorboard import SummaryWriter
import time
from utils.util import load_checkpoint
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from train_module.data_helper import save_tensor_as_image

def get_args():
	parser = argparse.ArgumentParser()
	# Train Parameter
	train = parser.add_argument_group('Train Option')
	train.add_argument('--img_path', type=list, default=[os.path.join('dataset', 'train_image_unbalanced'), os.path.join('dataset','aihub_unbalanced')])
	train.add_argument('--save_path', type=str, default=os.path.join('train_module','checkpoints'))
	train.add_argument('--split_rate', type=list, default=[0.8, 0.2])

	train.add_argument('--epochs', type=int, default=25)
	train.add_argument('--batch_size', type=int, default=80)
	train.add_argument('--model', choices=['vgg', 'cspvgg', 'inception'], type=str, default='vgg')
	train.add_argument('--learning_rate', type=float, default=0.003)
	train.add_argument('--print_train_step', type=int, default=100)
	train.add_argument('--saving_point_step', type=int, default=2500)
	train.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to resume training')

	# Model
	vgg_network = parser.add_argument_group(title='VGG Network Option')
	vgg_network.add_argument('--vgg_type', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg-s'], type=str,
							 default='vgg19')
	vgg_network.add_argument('--vgg_batch_norm', type=bool, default=True)
	return parser.parse_args()


class Trainer(object):
	def __init__(self, arguments):
		self.arguments = arguments
		info_paths = [os.path.join(img_path, 'metadata.json') for img_path in self.arguments.img_path]
		self.dataset_info = split_dataset(info_paths, split_rate=arguments.split_rate)

		print('\n===========Train set age classes ratio==============')
		print(f"<20대: {sum(d['age']<20 for d in self.dataset_info[0])}")
		print(f"20대: {sum(d['age']==20 for d in self.dataset_info[0])}")
		print(f"30대: {sum(d['age']==30 for d in self.dataset_info[0])}")
		print(f"40대: {sum(d['age']==40 for d in self.dataset_info[0])}")
		print(f"50대: {sum(d['age']==50 for d in self.dataset_info[0])}")
		print(f"60대: {sum(d['age']==60 for d in self.dataset_info[0])}")
		print(f">60대: {sum(d['age']>60 for d in self.dataset_info[0])}")
		print('===========Train set gender classes ratio==============')
		print(f"male: {sum(d['gender']=='male' for d in self.dataset_info[0])}")
		print(f"female: {sum(d['gender']=='female' for d in self.dataset_info[0])}")
		print('====================================================\n\n')


		print('\n===========Val set age classes ratio==============')
		print(f"<20대: {sum(d['age']<20 for d in self.dataset_info[1])}")
		print(f"20대: {sum(d['age']==20 for d in self.dataset_info[1])}")
		print(f"30대: {sum(d['age']==30 for d in self.dataset_info[1])}")
		print(f"40대: {sum(d['age']==40 for d in self.dataset_info[1])}")
		print(f"50대: {sum(d['age']==50 for d in self.dataset_info[1])}")
		print(f"60대: {sum(d['age']==60 for d in self.dataset_info[0])}")
		print(f">60대: {sum(d['age']>60 for d in self.dataset_info[0])}")
		print('====================================================')
		print('===========Val set gender classes ratio==============')
		print(f"male: {sum(d['gender']=='male' for d in self.dataset_info[1])}")
		print(f"female: {sum(d['gender']=='female' for d in self.dataset_info[1])}")
		print('====================================================\n\n')
		self.gender_model, self.age_model = self.get_model()       # Define FaceRecognition
		self.gender_branch_criterion = nn.BCELoss()
		# age_weights = torch.FloatTensor([148921,61181,8055,390,186]).to(device) ## this is frequence of age classes in dataset (from left <20, 20~35, 35~50, 50~65, >65)
		# age_weights = torch.nn.functional.normalize(age_weights, dim=0, p=1)
		# self.age_branch_criterion = nn.CrossEntropyLoss(weight=age_weights)
		#### when using softmax to predict categorical classes (age as categorical variable)
		# self.age_branch_criterion = nn.CrossEntropyLoss()
		# #### when using mse to estimating age (age as continuos variable)
		self.age_branch_criterion = nn.MSELoss()
		# self.age_branch_criterion = nn.L1Loss()

		self.gender_optimizer = opt.Adam(self.gender_model.parameters(),lr=self.arguments.learning_rate)
		self.age_optimizer = opt.Adam(self.age_model.parameters(),lr=self.arguments.learning_rate)

		self.model_name = None
		if self.arguments.model == 'vgg':
			self.model_name = self.arguments.model
		elif self.arguments.model == 'cspvgg':
			self.model_name = self.arguments.model
		elif self.arguments.model == 'inception':
			self.model_name = self.arguments.model

		self.writer = SummaryWriter('runs/model_{}-batch_{}-lr_{}'.format(
			self.model_name,
			self.arguments.batch_size,
			self.arguments.learning_rate
		))

	def train(self, s_epoch=0):
		# Train DataLoader
		train_loader = self.train_loader()
		# model & optimizer & loss
		self.gender_model.to(device)
		self.age_model.to(device)


		# Training
		total_it = 0
		val_total_loss, val_total_gender_loss, val_total_age_loss, val_total_gender_acc, val_total_age_acc = 0, 0, 0, 0, 0

		# Training Start...
		save=False
		for epoch in range(s_epoch, self.arguments.epochs):
			start = time.time()
			for i, (img,gender,age) in enumerate(train_loader):
				rand = random.randint(1,4)
				self.gender_model.train()
				self.age_model.train()
				# optimizer & back propagation
				self.gender_optimizer.zero_grad()
				self.age_optimizer.zero_grad()
				img,gender,age = self.pad_tensor((img, gender, age), self.arguments.batch_size)
				img, gender, age =img.to(device),gender.to(device),age.to(device)

				pred_gender = self.gender_model(img)
				pred_age = self.age_model(img)

				# print(f"pred_gender shape: {pred_gender.shape} -- real gender shape: {gender.shape}")
				# print(f"pred_age shape: {pred_age.shape} -- real age shape: {age.shape}")
				gender_loss =self.gender_branch_criterion(pred_gender, gender)
				age_loss =self.age_branch_criterion(pred_age.squeeze(), age)

				if rand == 2:
					print(f'gender_loss: {float(gender_loss)} -- Age loss: {float(age_loss)}')
				gender_loss.backward()
				age_loss.backward()
				self.gender_optimizer.step()
				self.age_optimizer.step()

				total_it += 1
				# Train 결과 출력
				if i % self.arguments.print_train_step == 0:
					# Console 출력
					# gender_out, age_out = torch.argmax(out_['gender'],dim=-1).tolist(), torch.argmax(out_['age'],dim=-1).tolist()
					# gender_tar, age_tar = torch.argmax(gender,dim=-1).tolist(), age.tolist()
					# print('\n================')
					# print(f'pred_gender: {gender_out}\nreal_gender: {gender_tar}')
					# print('================')
					# print(f'pred_age: {age_out}\nreal_age: {age_tar}')
					# print('================')
					# print(f'Epoch: {epoch} -- iter: {i} -- loss: {loss} ')
					# Tensorboard 출력
					self.writer.add_scalar('02.train_gender/loss', gender_loss, total_it)
					self.writer.add_scalar('03.train_age/loss', age_loss, total_it)

				# 모델 저장
				if i % self.arguments.saving_point_step==0:
					# Console 출력
					val_total_gender_acc, val_total_age_acc = self.val(save=save)
					save = False
					print('\n================================')
					print(f'Validation:')
					print(f'Gender acc: {val_total_gender_acc} --- Age acc: {val_total_age_acc}')
					print('================================\n')

					# Tensorboard 출력
					self.writer.add_scalar('02.val_gender/accuracy', val_total_gender_acc, total_it)
					self.writer.add_scalar('03.val_age/accuracy', val_total_age_acc, total_it)

					self.save_model(
						epochs=epoch,
						it=i,
						val_loss_accuracy={
							'gender_accuracy': val_total_gender_acc,
							'age_accuracy': val_total_age_acc
						}
					)



			print(f'Finished  {epoch} epochs: -- it takes {time.time()-start} seconds..')
		# save model at the last
		self.save_model(
			epochs=self.arguments.epochs,
			it=total_it,
			val_loss_accuracy={
				'gender_accuracy': val_total_gender_acc,
				'age_accuracy': val_total_age_acc
			}
		)
		val_total_gender_acc, val_total_age_acc = self.val(save=save)

		print('\n================================')
		print(f'Validation: Gender acc: {val_total_gender_acc} --- Age acc: {val_total_age_acc}')
		print('================================\n')
		# Tensorboard 출력
		self.writer.add_scalar('02.val_gender/accuracy', val_total_gender_acc, total_it)
		self.writer.add_scalar('03.val_age/accuracy', val_total_age_acc, total_it)

	def val(self,save=False):
		# Validation DataLoader
		print(f"\n=====================================================")
		print(f"Starting Validation..")
		print(f"=====================================================\n")
		val_loader = self.val_loader()
		total_images = len(val_loader) * 1 #### validation batch size set to 1
		total_gender_acc, total_age_acc = 0, 0
		self.gender_model.eval()
		self.age_model.eval()
		gender_correct_count, age_correct_count = 0,0

		with torch.no_grad():
			for img,gender,age in val_loader:
				img, gender, age = img.to(device), gender.to(device), age.to(device)
				rand=random.randint(1,3500)
				pred_gender = self.gender_model(img)
				pred_age = self.age_model(img)
				gender_target = torch.argmax(gender,dim=1).to(device)
				age_target = age.to(device)
				gender_out = torch.argmax(pred_gender, dim=1).item()
				age_out = pred_age.item()
				if save:
					temp = time.time()
					os.makedirs('val_images', exist_ok=True)
					img_ = img.squeeze(0)
					save_tensor_as_image(img_,f"val_images/{temp}_{int(torch.argmax(gender, dim=1))}_{int(age)}.png")

				if rand==2:
					# gender_tar, age_tar = torch.argmax(gender,dim=-1).tolist(), age.tolist()
					print(f"\n=====================================================")
					print(f"predicted gender {round(float(gender_out))}: {pred_gender}")
					print(f"predicted age {round(float(age_out))}: {pred_age}")
					print(f"Real: {round(float(gender_target))} -- {round(float(age_target))}")
					print(f"=====================================================\n")


				if round(float(gender_out)) == round(float(gender_target)):
					gender_correct_count+=1
				if round(float(age_out)) == round(float(age_target)):
					age_correct_count +=1
		# calculation average loss & accuracy
		gender_acc = 100 * gender_correct_count / total_images
		age_acc = 100 * age_correct_count / total_images
		print(f"validation gender correct count: {gender_correct_count}/{total_images}")
		print(f"validation age correct count: {age_correct_count}/{total_images}")

		return gender_acc, age_acc

	# def get_loss(self, out, tar):
	# 	"""
	# 	take predicted output and target then calculate the loss and (count of) accuracy
	# 	"""
	# 	gender_out, age_out = out['gender'].to(device), out['age'].to(device)
	# 	gender_tar, age_tar = tar['gender'].to(device), tar['age'].to(device)
	#
	# 	# calculation loss
	# 	gender_loss = self.gender_branch_criterion(input=gender_out,target=gender_tar)
	# 	age_loss = self.age_branch_criterion(input=age_out,target=age_tar)
	# 	return gender_loss, age_loss



	def get_accuracy(self, out, tar):
		"""
		take predicted output and target then calculate the loss and (count of) accuracy
		"""
		gender_out, age_out = out['gender'].to(device), out['age'].to(device)
		gender_tar, age_tar = tar['gender'].to(device), tar['age'].to(device)
		# calculation accuracy
		### because target gender is one hot tensor --> convert to int first
		gender_correct = torch.argmax(gender_tar,dim=-1).eq(torch.argmax(gender_out,dim=-1)).sum().to(torch.float32)
		### target age is int, do not need to convert to int
		age_indicate = age_out.argmax(dim=-1)
		age_correct = age_tar.eq(age_out).sum().to(torch.float32)
		return gender_correct, age_correct

	def train_loader(self) -> DataLoader:
		# train information.json List
		train_info = self.dataset_info[0]

		# train FaceImageDataset & DataLoader
		train_dataset = FaceImageDataset(info=train_info)

		train_data_loader = DataLoader(
			dataset=train_dataset, batch_size=self.arguments.batch_size,
			shuffle=True
		)
		return train_data_loader

	def val_loader(self) -> DataLoader:
		# validation information.json List
		val_info = self.dataset_info[1]
		# validation FaceImageDataset & DataLoader
		val_dataset = FaceImageDataset(info=val_info)
		val_data_loader = DataLoader(dataset=val_dataset, batch_size=1)
		return val_data_loader

	def get_model(self):
		model_name = self.arguments.model
		if model_name == 'cspvgg':
			from modules.vgg import CSP_VGG
			return CSP_VGG(vgg_type='vgg19')
			# from train_module.models.vgg import VGG
			# return VGG(**parameter)
		elif model_name == 'vgg':
			from modules.vgg import Gender_VGG, Age_VGG
			return Gender_VGG(vgg_type='vgg19'), Age_VGG(vgg_type='vgg19')
		elif model_name == 'inception':
			from modules.vgg import Inception_VGG
			return Inception_VGG(vgg_type='vgg19')

	def save_model(self, epochs, it, val_loss_accuracy):
		filename = f'{self.model_name}-epochs_{epochs}-step_{it}-gender_acc_{val_loss_accuracy["gender_accuracy"]}-age_acc_{val_loss_accuracy["age_accuracy"]}.pth'
		os.makedirs(self.arguments.save_path, exist_ok=True)
		filepath = os.path.join(self.arguments.save_path, filename)
		torch.save({
			'parameter': {
				'epoch': epochs,
				'iterator': it,
				'batch_size': self.arguments.batch_size,
				'learning_rate': self.arguments.learning_rate
			},
			'gender_model_weights': self.gender_model.state_dict(),
			'age_model_weights': self.age_model.state_dict(),
			'gender_optimizer_weights': self.gender_optimizer.state_dict(),
			'age_optimizer_weights': self.age_optimizer.state_dict(),
			'model_type': self.arguments.model
		}, filepath)


	def pad_tensor(self,tensors, size):
		"""
		args:
			tensors - tenintsors to pad
			size - the size to pad to
			dim - dimension to pad

		return:
			a new tensor padded to 'pad' in dimension 'dim'
		"""

		img, gender, age = tensors
		if len(img) < size:

			img_pad = torch.unsqueeze(img[-1],dim=0)
			age_pad = torch.unsqueeze(age[-1], dim=0)
			gender_pad = torch.unsqueeze(gender[-1], dim=0)

			for i in range(size-len(img)):
				img = torch.cat((img, img_pad), dim=0)
				age = torch.cat((age, age_pad), dim=0)
				gender = torch.cat((gender, gender_pad), dim=0)
		return img, gender, age
if __name__ == '__main__':
	args = get_args()
	trainer = Trainer(args)
	s_epoch=0
	if args.checkpoint is not None:
		model, epoch = load_checkpoint(args.checkpoint)
		trainer.model = model
		trainer.model.train()
	print(f'start epoch {s_epoch}')
	trainer.train(s_epoch=s_epoch)

