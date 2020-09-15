# -*- coding:utf-8 -*-
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from train_module.data_helper import FaceImageDataset, split_dataset
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
	parser = argparse.ArgumentParser()
	# Train Parameter
	train = parser.add_argument_group('Train Option')
	train.add_argument('--img_path', type=str, default=os.path.join('dataset', 'train_image_unbalanced'))
	train.add_argument('--info_path', type=str, default=os.path.join('dataset', 'metadata_unbalanced.json'))
	train.add_argument('--save_path', type=str, default=os.path.join('train_module','store'))
	train.add_argument('--split_rate', type=list, default=[0.8, 0.1, 0.1])

	train.add_argument('--epochs', type=int, default=25)
	train.add_argument('--batch_size', type=int, default=200)
	train.add_argument('--model', choices=['vgg', 'fr_net'], type=str, default='vgg')
	train.add_argument('--learning_rate', type=float, default=0.01)
	train.add_argument('--print_train_step', type=int, default=100)
	train.add_argument('--print_val_step', type=int, default=1000)
	train.add_argument('--saving_point_step', type=int, default=2500)

	# Model
	vgg_network = parser.add_argument_group(title='VGG Network Option')
	vgg_network.add_argument('--vgg_type', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg-s'], type=str,
							 default='vgg19')
	vgg_network.add_argument('--vgg_batch_norm', type=bool, default=True)
	return parser.parse_args()


class Trainer(object):
	def __init__(self, arguments):
		self.arguments = arguments
		self.dataset_info = split_dataset(arguments.info_path, split_rate=arguments.split_rate)
		self.model = self.get_model()       # Define FaceRecognition
		self.sex_branch_criterion = nn.CrossEntropyLoss()
		age_weights = torch.FloatTensor([148921,61181,8055,390,186]).to(device) ## this is frequence of age classes in dataset (from left <20, 20~35, 35~50, 50~65, >65)
		age_weights = torch.nn.functional.normalize(age_weights, dim=0, p=1)


		self.age_branch_criterion = nn.CrossEntropyLoss(weight=age_weights)
		# self.age_branch_criterion = nn.CrossEntropyLoss()

		self.optimizer = opt.Adam(
			self.model.parameters(),
			lr=self.arguments.learning_rate
		)

		self.model_name = None
		if self.arguments.model == 'vgg':
			self.model_name = self.arguments.vgg_type
		elif self.arguments.model == 'fr_net':
			self.model_name = self.arguments.model

		self.writer = SummaryWriter('runs/model_{}-batch_{}-lr_{}'.format(
			self.model_name,
			self.arguments.batch_size,
			self.arguments.learning_rate
		))

	def train(self):
		# Train DataLoader
		train_loader = self.train_loader()

		# model & optimizer & loss
		self.model.to(device)
		self.model.train()

		# Training
		total_it = 0
		loss, out = None, None
		val_total_loss, val_total_sex_loss, val_total_age_loss, val_total_sex_acc, val_total_age_acc = 0, 0, 0, 0, 0

		# Training Start...
		for epoch in range(self.arguments.epochs):
			for i, data in enumerate(train_loader):
				data = self.pad_tensor(data, self.arguments.batch_size)
				out = self.model(data['img'])
				# print(f'model output: {out}')
				del data['img']

				out = self.get_accuracy_loss(out=out, tar=data)
				# print(f"gender loss: {out['loss']['sex']}")
				# print(f"age loss: {out['loss']['age']}")


				loss = torch.sqrt(torch.square(out['loss']['sex']) + torch.square(out['loss']['age']))

				# Train 결과 출력
				if i % self.arguments.print_train_step == 0:
					# Console 출력
					self.print_fn(mode='Train',
								  epoch=epoch, it=i,
								  total_loss=loss,
								  sex_loss=out['loss']['sex'].item(),
								  age_loss=out['loss']['age'].item(),
								  sex_accuracy=out['accuracy']['sex'].item(),
								  age_accuracy=out['accuracy']['age'].item())

					# Tensorboard 출력
					self.writer.add_scalar('01.train/loss', loss.item(), total_it)
					self.writer.add_scalar('02.train_sex/loss', out['loss']['sex'].item(), total_it)
					self.writer.add_scalar('02.train_sex/accuracy', out['accuracy']['sex'].item(), total_it)
					self.writer.add_scalar('03.train_age/loss', out['loss']['age'].item(), total_it)
					self.writer.add_scalar('03.train_age/accuracy', out['accuracy']['age'].item(), total_it)

				# # Validation 결과 출력
				# if i % self.arguments.print_val_step == 0:
				# 	# Console 출력
				# 	val_total_loss, val_total_sex_loss, val_total_age_loss, val_total_sex_acc, val_total_age_acc = \
				# 		self.val()
				# 	self.print_fn(mode=' Val ',
				# 				  epoch=epoch, it=i,
				# 				  total_loss=val_total_loss,
				# 				  sex_loss=val_total_sex_loss, age_loss=val_total_age_loss,
				# 				  sex_accuracy=val_total_sex_acc, age_accuracy=val_total_age_acc)
				#
				# 	# Tensorboard 출력
				# 	self.writer.add_scalar('01.val/loss', val_total_loss, total_it)
				# 	self.writer.add_scalar('02.val_sex/loss', val_total_sex_loss, total_it)
				# 	self.writer.add_scalar('02.val_sex/accuracy', val_total_sex_acc, total_it)
				# 	self.writer.add_scalar('03.val_age/loss', val_total_age_loss, total_it)
				# 	self.writer.add_scalar('03.val_age/accuracy', val_total_age_acc, total_it)

				# 모델 저장
				if i == 0:
					# Console 출력
					val_total_loss, val_total_sex_loss, val_total_age_loss, val_total_sex_acc, val_total_age_acc = self.val()
					self.print_fn(mode=' Val ',
								  epoch=epoch, it=i,
								  total_loss=val_total_loss,
								  sex_loss=val_total_sex_loss, age_loss=val_total_age_loss,
								  sex_accuracy=val_total_sex_acc, age_accuracy=val_total_age_acc)

					# Tensorboard 출력
					self.writer.add_scalar('01.val/loss', val_total_loss, total_it)
					self.writer.add_scalar('02.val_sex/loss', val_total_sex_loss, total_it)
					self.writer.add_scalar('02.val_sex/accuracy', val_total_sex_acc, total_it)
					self.writer.add_scalar('03.val_age/loss', val_total_age_loss, total_it)
					self.writer.add_scalar('03.val_age/accuracy', val_total_age_acc, total_it)

					self.save_model(
						epochs=epoch,
						it=i,
						train_loss_accuracy={
							'total_loss': loss,
							'sex_loss': out['loss']['sex'].item(),
							'age_loss': out['loss']['age'].item(),
							'sex_accuracy': out['accuracy']['sex'].item(),
							'age_accuracy': out['accuracy']['age'].item()
						},
						val_loss_accuracy={
							'total_loss': val_total_loss,
							'sex_loss': val_total_sex_loss,
							'age_loss': val_total_age_loss,
							'sex_accuracy': val_total_sex_acc,
							'age_accuracy': val_total_age_acc
						}
					)

				# optimizer & back propagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_it += 1

		# save model at the last
		self.save_model(
			epochs=self.arguments.epochs,
			it=total_it,
			train_loss_accuracy={
				'total_loss': loss,
				'sex_loss': out['loss']['sex'].item(),
				'age_loss': out['loss']['age'].item(),
				'sex_accuracy': out['accuracy']['sex'].item(),
				'age_accuracy': out['accuracy']['age'].item()
			},
			val_loss_accuracy={
				'total_loss': val_total_loss,
				'sex_loss': val_total_sex_loss,
				'age_loss': val_total_age_loss,
				'sex_accuracy': val_total_sex_acc,
				'age_accuracy': val_total_age_acc
			}
		)
		val_total_loss, val_total_sex_loss, val_total_age_loss, val_total_sex_acc, val_total_age_acc = self.val()

		self.print_fn(mode=' Val ',
					  epoch=epoch, it=total_it,
					  total_loss=val_total_loss,
					  sex_loss=val_total_sex_loss, age_loss=val_total_age_loss,
					  sex_accuracy=val_total_sex_acc, age_accuracy=val_total_age_acc)
		# Tensorboard 출력
		self.writer.add_scalar('01.val/loss', val_total_loss, total_it)
		self.writer.add_scalar('02.val_sex/loss', val_total_sex_loss, total_it)
		self.writer.add_scalar('02.val_sex/accuracy', val_total_sex_acc, total_it)
		self.writer.add_scalar('03.val_age/loss', val_total_age_loss, total_it)
		self.writer.add_scalar('03.val_age/accuracy', val_total_age_acc, total_it)

	def val(self):
		# Validation DataLoader
		val_loader = self.val_loader()
		total_loss = 0
		total_sex_loss, total_age_loss = 0, 0
		total_sex_acc, total_age_acc = 0, 0

		count = 0
		with torch.no_grad():
			for data in val_loader:
				data = self.pad_tensor(data, self.arguments.batch_size)
				out = self.model(data['img'])
				del data['img']

				out = self.get_accuracy_loss(out=out, tar=data)
				sex_loss, age_loss = out['loss']['sex'].item(), out['loss']['age'].item()
				# calculation loss & accuracy
				total_loss += sex_loss + age_loss
				total_sex_loss += sex_loss
				total_age_loss += age_loss
				total_sex_acc += out['accuracy']['sex'].item()
				total_age_acc += out['accuracy']['age'].item()
				count += 1

		# calculation average loss & accuracy
		total_loss /= count
		total_sex_loss /= count
		total_age_loss /= count
		total_age_acc /= count
		total_sex_acc /= count
		return total_loss, total_sex_loss, total_age_loss, total_sex_acc, total_age_acc

	def get_accuracy_loss(self, out, tar):
		sex_out, age_out = out['sex'].to(device), out['age'].to(device)
		sex_tar, age_tar = tar['sex'].to(device), tar['age'].to(device)

		# calculation loss
		sex_loss = self.sex_branch_criterion(input=sex_out,target=sex_tar
		)
		age_loss = self.age_branch_criterion(input=age_out,target=age_tar)

		# calculation accuracy
		sex_indicate = sex_out.argmax(dim=-1)
		sex_accuracy = sex_tar.eq(sex_indicate).sum().to(torch.float32) / sex_indicate.size()[0]
		age_indicate = age_out.argmax(dim=-1)
		age_accuracy = age_tar.eq(age_indicate).sum().to(torch.float32) / age_indicate.size()[0]

		return {
			'loss': {'sex': sex_loss, 'age': age_loss},
			'accuracy': {'sex': sex_accuracy, 'age': age_accuracy}
		}

	def train_loader(self) -> DataLoader:
		# train information.json List
		train_info = self.dataset_info[0]

		# train FaceImageDataset & DataLoader
		train_dataset = FaceImageDataset(
			root=self.arguments.img_path, info=train_info,
			mode='train', device=device
		)

		train_data_loader = DataLoader(
			dataset=train_dataset, batch_size=self.arguments.batch_size,
			shuffle=True
		)
		return train_data_loader

	def val_loader(self) -> DataLoader:
		# validation information.json List
		val_info = self.dataset_info[2]

		# validation FaceImageDataset & DataLoader
		val_dataset = FaceImageDataset(
			root=self.arguments.img_path, info=val_info,
			mode='validation', device=device
		)

		val_data_loader = DataLoader(dataset=val_dataset, batch_size=self.arguments.batch_size)
		return val_data_loader

	def get_model(self):
		parameter = self.get_parameter()
		model_name = self.arguments.model
		if model_name == 'vgg':
			from train_module.models.vgg import VGG
			return VGG(**parameter)
		elif model_name == 'fr_net':
			from train_module.models.frnet import FRNet
			return FRNet()

	def save_model(self, epochs, it, train_loss_accuracy, val_loss_accuracy):

		filename = '{0}-epochs_{1}-step_{2}-gender_accuracy_{3}.pth'.format(self.model_name, epochs, it, val_loss_accuracy['sex_accuracy'])
		filepath = os.path.join(self.arguments.save_path, filename)
		torch.save({
			'parameter': {
				'epoch': epochs,
				'iterator': it,
				'batch_size': self.arguments.batch_size,
				'learning_rate': self.arguments.learning_rate
			},
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'model_type': self.arguments.model,
			'model_parameter': self.get_parameter(),
			'train_loss_accuracy': train_loss_accuracy,
			'val_loss_accuracy': val_loss_accuracy
		}, filepath)

	def get_parameter(self):
		model_name = self.arguments.model
		parameter = {}
		if model_name == 'vgg':
			parameter = {
				'vgg_type': self.arguments.vgg_type,
				'batch_norm': self.arguments.vgg_batch_norm
			}
		elif model_name == 'inception':
			parameter = {
				'type': self.arguments.inception_type
			}
		return parameter

	@staticmethod
	def print_fn(epoch, it, total_loss, sex_loss, age_loss, sex_accuracy, age_accuracy, mode='Train'):
		print(
			'[ {0} ] => '
			'[Epoch] : {1:3d}  [Iter] : {2:3d}  [Total Loss] : {3:9.4f}  '
			'[Sex Loss] : {4:9.4f}  [Sex Accuracy] : {6:9.4f}  '
			'[Age Loss] : {5:9.4f}  [Age Accuracy] : {7:9.4f}'.format(
					mode,
					epoch, it, total_loss,
					sex_loss, age_loss,
					sex_accuracy, age_accuracy)
		)


	def pad_tensor(self,tensors, size):
		"""
		args:
			tensors - tenintsors to pad
			size - the size to pad to
			dim - dimension to pad

		return:
			a new tensor padded to 'pad' in dimension 'dim'
		"""


		if len(tensors['img']) < size:

			img_pad = torch.unsqueeze(tensors['img'][-1],dim=0)
			age_pad = torch.unsqueeze(tensors['age'][-1], dim=0)
			sex_pad = torch.unsqueeze(tensors['sex'][-1], dim=0)

			for i in range(size-len(tensors['img'])):
				tensors['img'] = torch.cat((tensors['img'], img_pad), dim=0)
				tensors['age'] = torch.cat((tensors['age'], age_pad), dim=0)
				tensors['sex'] = torch.cat((tensors['sex'], sex_pad), dim=0)
		return tensors
if __name__ == '__main__':
	args = get_args()
	Trainer(args).train()

