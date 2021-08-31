# -*- coding:utf-8 -*-
import os
import random

import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_module.data_helper import FaceImageDataset, split_dataset
from torch.utils.tensorboard import SummaryWriter
import time
from utils.util import load_checkpoint
from train_module.data_helper import save_tensor_as_image
torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)


def get_args():
	parser = argparse.ArgumentParser()
	# Train Parameter
	train = parser.add_argument_group('Train Option')
	# train.add_argument('--img_path', type=list, default=[os.path.join('dataset', 'train_image_unbalanced')])
	train.add_argument('--img-path', type=list, default=[os.path.join('dataset', 'train_image_unbalanced'), os.path.join('dataset','aihub_unbalanced')])
	train.add_argument('--save-path', type=str, default=os.path.join('train_module','checkpoints'))
	train.add_argument('--split-rate', type=list, default=[0.8, 0.2])
	train.add_argument('--age-balance', type=int, default=None, help='number of each age class data to be limited, None means use all of them')
	train.add_argument('--epochs', type=int, default=25)
	train.add_argument('--device', type=int, default=0)
	train.add_argument('--type', type=int, default=0, help='0: gender, 1: age')
	train.add_argument('--batch-size', type=int, default=80)
	train.add_argument('--model', choices=['vgg', 'cspvgg', 'inception'], type=str, default='vgg')
	train.add_argument('--learning-rate', type=float, default=0.0025)
	train.add_argument('--print-train-step', type=int, default=100)
	train.add_argument('--saving-point-step', type=int, default=2500)
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
		self.dataset_info = split_dataset(info_paths, split_rate=arguments.split_rate, age_balance_count=self.arguments.age_balance)

		self.type = self.arguments.type
		if self.type==1:
			print('\n===========Train set age classes ratio==============')
			print(f"<20대: {sum(d['age']==10 for d in self.dataset_info[0])}")
			print(f"20대: {sum(d['age']==20 for d in self.dataset_info[0])}")
			print(f"30대: {sum(d['age']==30 for d in self.dataset_info[0])}")
			print(f"40대: {sum(d['age']==40 for d in self.dataset_info[0])}")
			print(f">50대: {sum(d['age']==50 for d in self.dataset_info[0])}")
		else:
			print('===========Train set gender classes ratio==============')
			print(f"male: {sum(d['gender']=='male' for d in self.dataset_info[0])}")
			print(f"female: {sum(d['gender']=='female' for d in self.dataset_info[0])}")
			print('====================================================\n\n')

		if self.type == 1:
			print('\n===========Val set age classes ratio==============')

			print(f"<20대: {sum(d['age']==10 for d in self.dataset_info[1])}")
			print(f"20대: {sum(d['age']==20 for d in self.dataset_info[1])}")
			print(f"30대: {sum(d['age']==30 for d in self.dataset_info[1])}")
			print(f"40대: {sum(d['age']==40 for d in self.dataset_info[1])}")
			print(f">50대: {sum(d['age']==50 for d in self.dataset_info[1])}")
			print('====================================================')
		else:
			print('===========Val set gender classes ratio==============')
			print(f"male: {sum(d['gender']=='male' for d in self.dataset_info[1])}")
			print(f"female: {sum(d['gender']=='female' for d in self.dataset_info[1])}")
			print('====================================================\n\n')
		self.model = self.get_model()       # Define FaceRecognition
		self.criterion = nn.MSELoss()
		# age_weights = torch.FloatTensor([148921,61181,8055,390,186]).to(device) ## this is frequence of age classes in dataset (from left <20, 20~35, 35~50, 50~65, >65)
		# age_weights = torch.nn.functional.normalize(age_weights, dim=0, p=1)
		# self.age_branch_criterion = nn.CrossEntropyLoss(weight=age_weights)
		#### when using softmax to predict categorical classes (age as categorical variable)
		# self.age_branch_criterion = nn.CrossEntropyLoss()
		# #### when using mse to estimating age (age as continuos variable)
		# self.age_branch_criterion = nn.MSELoss()
		# self.age_branch_criterion = nn.L1Loss()

		self.optimizer = opt.Adam(self.model.parameters(),lr=self.arguments.learning_rate)
		# self.age_optimizer = opt.Adam(self.age_model.parameters(),lr=self.arguments.learning_rate)

		self.model_name = 'vgg'

		self.writer = SummaryWriter('runs/model_{}-batch_{}-lr_{}'.format(
			self.model_name,
			self.arguments.batch_size,
			self.arguments.learning_rate
		))

	def train(self, s_epoch=0):
		# Train DataLoader
		train_loader = self.train_loader()
		# model & optimizer & loss
		self.model.to(device)
		# self.age_model.to(device)


		# Training
		total_it = 0
		val_total_loss, val_total_gender_loss, val_total_age_loss, val_total_gender_acc, val_total_age_acc = 0, 0, 0, 0, 0

		# Training Start...
		save=False
		best_acc = 0

		for epoch in range(s_epoch, self.arguments.epochs):

			start = time.time()
			train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
			for i, (img,gender,age) in train_progress_bar:
				target = gender if self.type==0 else age
				self.model.train()
				# self.age_model.train()
				# optimizer & back propagation
				self.optimizer.zero_grad()
				# self.age_optimizer.zero_grad()
				# img, padded_target = self.pad_tensor((img, target), self.arguments.batch_size)
				# print(img.shape)
				img, target = img.to(device), target.to(device)
				# print(self.model)
				pred = self.model(img)
				# pred_age = self.age_model(img)
				# print(f'padded_target {padded_target}')
				# print(f'pred {pred}')

				# print(f"pred_gender shape: {pred_gender.shape} -- real gender shape: {gender.shape}")
				# print(f"pred_age shape: {pred_age.shape} -- real age shape: {age.shape}")
				if self.type==0:
					loss =self.criterion(pred, target)
				else:
					loss =self.criterion(pred.squeeze(), target)
				if i%50==0:
					if self.type==0:
						print(f'gender_loss: {loss}')
					else:
						print(f'age_loss: {loss}')

				loss.backward()
				# age_loss.backward()
				self.optimizer.step()
				# self.age_optimizer.step()

				total_it += 1
				# Train 결과 출력
				if i % self.arguments.print_train_step == 0:
					if self.type==0:
						self.writer.add_scalar('02.train_gender/loss', loss, total_it)
					else:
						self.writer.add_scalar('03.train_age/loss', loss, total_it)
			# 모델 저장
			# Console 출력
			val_total_acc= self.val(save=save)
			print('\n================================')
			print(f'Validation:')
			print(f'val_total_acc: {val_total_acc}')
			print('================================\n')

			# Tensorboard 출력
			if self.type==0:
				self.writer.add_scalar('02.val_gender/accuracy', val_total_acc, total_it)
			else:
				self.writer.add_scalar('03.val_age/accuracy', val_total_acc, total_it)

			### save model
			type = 'gender' if self.type == 0 else 'age'
			filename = f'epochs_{epoch}-{type}_acc_{val_total_acc}.pth'
			self.save_model(
				epochs=self.arguments.epochs,
				filename=filename
			)
			#### save best model
			if val_total_acc >= best_acc:
				best_acc = val_total_acc
				filename = f'{type}-best.pth'
				self.save_model(epochs=epoch, filename=filename)

			print(f'Finished  epoch {epoch}: -- it takes {time.time()-start} seconds..')


		val_total_acc = self.val(save=save)
		print(f'\n=============type: {"Gender" if self.type==0 else "Age"}===================')
		print(f'Validation acc: {val_total_acc}')
		print('================================\n')

		# Tensorboard 출력
		if self.type==0:
			self.writer.add_scalar('02.val_gender/accuracy', val_total_acc, total_it)
		else:
			self.writer.add_scalar('03.val_age/accuracy', val_total_acc, total_it)
		# save last model
		type = 'gender' if self.type==0 else 'age'
		filename = f'epochs_{self.arguments.epochs}-{type}_acc_{val_total_acc}.pth'
		self.save_model(
			epochs=self.arguments.epochs,
			filename=filename
		)

	def val(self,save=False):
		# Validation DataLoader
		print(f"\n=====================================================")
		print(f"Starting Validation..")
		print(f"=====================================================\n")
		val_loader = self.val_loader()
		total_images = len(val_loader) * 1 #### validation batch size set to 1
		total_gender_acc, total_age_acc = 0, 0
		self.model.eval()
		# self.age_model.eval()
		correct_count=0
		val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))

		with torch.no_grad():
			for idx, (img,gender,age) in val_progress_bar:
				img, gender, age = img.to(device), gender.to(device), age.to(device)
				if self.type==0:
					target = gender
					pred = self.model(img)
					# pred_age = self.age_model(img)
					target = torch.argmax(target, dim=1).to(device)
					out = torch.argmax(pred, dim=1).item()

				else:
					target = age
					pred = self.model(img)
					target = target.to(device)
					out = pred.item()

				if save:
					temp = time.time()
					os.makedirs('val_images', exist_ok=True)
					img_ = img.squeeze(0)
					save_tensor_as_image(img_,f"val_images/{temp}_{int(target)}.png")
				if idx%500==0:
					print('')
					print(f'pred: {round(float(out))} -- target: {round(float(target.cpu()))}')
					print('')
				if round(float(out)) == round(float(target.cpu())):
					correct_count+=1

		# calculation average loss & accuracy
		acc = 100 * correct_count / total_images
		print(f"validation correct count: {correct_count}/{total_images}")

		return acc

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

	# def get_model(self):
	# 	from modules.vgg import Gender_VGG, Age_VGG
	# 	if self.type==0:
	# 		return Gender_VGG(vgg_type='vgg19')
	# 	else:
	# 		return Age_VGG(vgg_type='vgg19')

	def get_model(self):
		from modules.vgg import Gender_New, Age_New
		if self.type==0:
			return Gender_New()
		else:
			return Age_New()


	def save_model(self, epochs, filename):
		os.makedirs(self.arguments.save_path, exist_ok=True)
		filepath = os.path.join(self.arguments.save_path, filename)
		torch.save({
			'parameter': {
				'epoch': epochs,
				'batch_size': self.arguments.batch_size,
				'learning_rate': self.arguments.learning_rate
			},
			'model_weights': self.model.state_dict(),
			'optimizer_weights': self.optimizer.state_dict(),
			'model_type': self.arguments.model,
			'type': self.type ### gender or age model?
		}, filepath)

	@staticmethod
	def pad_tensor(tensors, size, dim=0):
		"""
		args:
			tensors - tenintsors to pad
			size - the size to pad to
			dim - dimension to pad

		return:
			a new tensor padded to 'pad' in dimension 'dim'
		"""

		if tensors.size(dim) < size:
			tensors_pad = torch.unsqueeze(tensors[-1], dim=0)

			for i in range(size-len(tensors.size(dim))):
				img = torch.cat((tensors, tensors_pad), dim=0)

		return tensors

if __name__ == '__main__':
	args = get_args()
	device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
	trainer = Trainer(args)
	s_epoch=0
	if args.checkpoint is not None:
		model, epoch = load_checkpoint(args.checkpoint)
		trainer.model = model
		trainer.model.train()
	print(f'start epoch {s_epoch}')
	trainer.train(s_epoch=s_epoch)

