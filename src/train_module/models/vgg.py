# -*- coding:utf-8 -*-
import torch.nn as nn
from modules.classify import Classify, Classify_
import torch

cfg = {
	'vgg-s': [64,     'M', 128,      'M', 256, 256,           'M'],
	'vgg11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
	'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
	'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
	'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
age_classes = 5


class VGG(nn.Module):
	def __init__(self, vgg_type, batch_norm=True):
		super().__init__()
		self.vgg_type = vgg_type
		self.batch_norm = batch_norm
		self.cnn = self.cnn_layers()
		input_size = 2048
		if vgg_type=='vgg-s':
			input_size = input_size * 8
		self.classify = Classify_(input_size=input_size, age_classes=age_classes)
	def forward(self, inputs):
		cnn_output = self.cnn(inputs)
		x = cnn_output.view(cnn_output.size(0), -1)
		x = self.classify(x)
		return x

	def cnn_layers(self):
		layers = []
		input_channel = 3 ### RGB image
		#### vgg 19 architecture: 2 conv_64 + max pooling + 2 conv_128 + max_pooling + 4 conv_256 + max_pooling + (4 conv_512 + max_pooling) * 2
		for output_channel in cfg[self.vgg_type]:
			if output_channel == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				continue
			layers += [nn.Conv2d(in_channels=input_channel, out_channels=output_channel,kernel_size=(3, 3), padding=1)]
			if self.batch_norm:
				layers += [nn.BatchNorm2d(output_channel)]
			layers += [nn.ReLU(inplace=True)]
			input_channel = output_channel
		return nn.Sequential(*layers)


class Inception_VGG(nn.Module):
	def __init__(self, vgg_type, batch_norm=True):
		super().__init__()
		self.vgg_type = vgg_type
		self.batch_norm = batch_norm
		self.cnn = self.cnn_layers()
		input_size = 2048
		self.classify = Classify_(input_size=input_size, age_classes=age_classes)

	def forward(self, inputs):
		cnn_output = self.cnn(inputs)
		x = cnn_output.view(cnn_output.size(0), -1)
		x = self.classify(x)
		return x

	def cnn_layers(self):
		layers = []
		input_channel = 3 ### RGB image
		#### vgg 19 architecture: 2 conv_64 + max pooling + 2 conv_128 + max_pooling + 4 conv_256 + max_pooling + (4 conv_512 + max_pooling) * 2
		for output_channel in cfg[self.vgg_type]:
			if output_channel == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
				continue
			layers += [
				CSP_Inception(in_channels=input_channel, out_channels=output_channel)]
			if self.batch_norm:
				layers += [nn.BatchNorm2d(output_channel)]
			layers += [nn.ReLU(inplace=True)]
			input_channel = output_channel
		return nn.Sequential(*layers)


class Conv2d(nn.Module):
	'''
	Usual 2D convolutional neural network. Included the batch normalization and activation function.
	If the batch normalization is not necessary, use fuseforward instead of forward function.
	'''
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, activation=None, w_init_gain='linear'):  # ch_in, ch_out, kernel, stride, padding, groups
		super(Conv2d, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
		self.bn = torch.nn.BatchNorm2d(out_channels)
		activation = activation.strip().replace(' ','').lower() if activation is not None else activation
		assert activation in ['relu','silu','leakyrelu','tank','sigmoid','relu6',None],"activation function must be one of ['relu','relu6','silu','leakyrelu','tank','sigmoid']"
		if activation =='relu':
			self.activation = torch.nn.ReLU()
		elif activation == 'tanh':
			self.activation = torch.nn.Tanh()
		elif activation=='leakyrelu':
			self.activation = torch.nn.LeakyReLU()
		elif activation=='sigmoid':
			self.activation = torch.nn.Sigmoid()
		elif activation=='relu6':
			self.activation = torch.nn.ReLU6()
		else:
			self.activation = None
		### initialized model weights
		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))


	def forward(self, x):
		'''
		forward the process.
		Parameters
		----------
		x input of model

		Returns
		-------
		output of model
		'''
		if self.activation is not None:
			return self.activation(self.bn(self.conv(x)))
		else:
			return self.bn(self.conv(x))


	def fuseforward(self, x):
		if self.activation is not None:
			return self.activation(self.conv(x))
		else:
			return self.conv(x)





class Inception(torch.nn.Module):
	def __init__(self, in_channels, bottneck_out_channel, conv_out_channels=32):
		super(Inception, self).__init__()
		self.conv1d_10 = Conv2d(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1) ### padding = kernel size//2
		self.conv1d_20 = Conv2d(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=5, stride=1, padding=2) ### padding = kernel size//2
		self.conv1d_40 = Conv2d(in_channels=bottneck_out_channel, out_channels=conv_out_channels, kernel_size=11, stride=1, padding=5) ### padding = kernel size//2
		#### residual_conv and bottleneck convolution must match the inputs shape [batchsize, in_channel, with, height]
		self.bottleneck = Conv2d(in_channels=in_channels, out_channels=bottneck_out_channel, kernel_size=1, stride=1)
		self.residual_conv = Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=1, stride=1)
		self.max_pooling = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.batch_norm = torch.nn.BatchNorm2d(conv_out_channels*4)

	def forward(self, inputs):
		# print(f'inputs {inputs.shape}')
		pool_out = self.max_pooling(inputs)
		# print(f'pool_out {pool_out.shape}')

		residual_out = self.residual_conv(pool_out)
		# print(f'residual_out {residual_out.shape}')

		bottleneck_output = self.bottleneck(inputs)
		# print(f'bottleneck_output {bottleneck_output.shape}')

		conv_10_out = self.conv1d_10(bottleneck_output)
		# print(f'conv_10_out {conv_10_out.shape}')

		conv_20_out = self.conv1d_20(bottleneck_output)
		# print(f'conv_20_out {conv_20_out.shape}')

		conv_40_out = self.conv1d_40(bottleneck_output)
		# print(f'conv_40_out {conv_40_out.shape}')

		conv_outs = torch.cat((conv_10_out,conv_20_out,conv_40_out,residual_out), dim=1)
		output = self.batch_norm(conv_outs)
		# print(f'output {output.shape}')

		return output


class CSP_Inception(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(CSP_Inception, self).__init__()
		self.in_channels=in_channels
		self.out_channels=out_channels
		# self.middle_channels = int(in_channels / 3) + (in_channels % 3 > 0)
		self.middle_channels = 1
		self.inception = Inception(in_channels=self.middle_channels, bottneck_out_channel=(self.middle_channels%2)+(self.middle_channels//2), conv_out_channels=in_channels - self.middle_channels)
		self.transistion = Conv2d(in_channels=(in_channels - self.middle_channels)*5, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

	def forward(self,inputs):
		inception_input, res_input = torch.split(inputs, [self.middle_channels, self.in_channels-self.middle_channels], dim=1) ### split at channels dimension
		# print(f'inception_input {inception_input.shape}')
		# print(f'res_input {res_input.shape}')
		inception_out = self.inception(inception_input)
		# print(f'inception_out {inception_out.shape}')
		csp_out = torch.cat((inception_out,res_input), dim=1)
		# print(f'csp_out {csp_out.shape}')
		csp_out = self.transistion(csp_out)
		# print(f'csp_out {csp_out.shape}')
		return csp_out
#
# #
# #
# # a = CSP_Inception(in_channels=14, out_channels=16)
# tensor = torch.zeros((32,3,64,64), dtype=torch.float).fill_(1)
# a = vgg(tensor)
# #
# # # a_1, a_2 = torch.split(tensor, [1,32-1],dim=1)
# # c = a(tensor)
# #

class CSP_Conv2d(nn.Module):
	'''
	Usual 2D convolutional neural network. Included the batch normalization and activation function.
	If the batch normalization is not necessary, use fuseforward instead of forward function.
	'''
	def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, activation=None, w_init_gain='linear'):  # ch_in, ch_out, kernel, stride, padding, groups
		super(CSP_Conv2d, self).__init__()
		self.middle_channels = int(in_channels / 3) + (in_channels % 3 > 0)
		self.conv = torch.nn.Conv2d(in_channels = self.middle_channels, out_channels=out_channels//2, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
		self.bn = torch.nn.BatchNorm2d(out_channels)
		activation = activation.strip().replace(' ','').lower() if activation is not None else activation
		assert activation in ['relu','silu','leakyrelu','tank','sigmoid','relu6',None],"activation function must be one of ['relu','relu6','silu','leakyrelu','tank','sigmoid']"
		if activation =='relu':
			self.activation = torch.nn.ReLU()
		elif activation == 'tanh':
			self.activation = torch.nn.Tanh()
		elif activation=='leakyrelu':
			self.activation = torch.nn.LeakyReLU()
		elif activation=='sigmoid':
			self.activation = torch.nn.Sigmoid()
		elif activation=='relu6':
			self.activation = torch.nn.ReLU6()
		else:
			self.activation = None
		### initialized model weights
		torch.nn.init.xavier_uniform_(
			self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))



	def forward(self, x):
		'''
		forward the process.
		Parameters
		----------
		x input of model

		Returns
		-------
		output of model
		'''

		csp_x, residual_x = torch.split(x, self.middle_channels, dim=1)
		cnn_out = self.conv(csp_x)

		if self.activation is not None:

			return self.activation(self.bn(self.conv(x)))
		else:
			return self.bn(self.conv(x))


	def fuseforward(self, x):
		if self.activation is not None:
			return self.activation(self.conv(x))
		else:
			return self.conv(x)


class CSP_VGG(nn.Module):
	def __init__(self, vgg_type, batch_norm=True):
		super().__init__()
		self.vgg_type = vgg_type
		self.batch_norm = batch_norm
		self.extend_cnn = Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
		self.middle_channels = int(32 / 2) + (32 % 2 > 0) ### 32 is from extend_cnn layer channels. Here we wanna split input to 2 tensor by channels
		self.transition_cnn = Conv2d(in_channels=512, out_channels=self.middle_channels, kernel_size=1, stride=1, padding=0) ### inchannels = 512 becase vgg last layer channels is 512
		self.max_pool = nn.MaxPool2d(kernel_size=64//8) ### 64 is input image size
		input_size = 2048
		if vgg_type=='vgg-s':
			input_size = input_size * 8
		self.classify = Classify_(input_size=input_size, age_classes=age_classes)
		self.vgg = self.cnn_layers(input_channel=self.middle_channels)




	def forward(self, inputs):
		# print(f'vgg.py 150 -- input shape: {inputs.shape}')
		### pass input goes through extend layer to expand channels
		extended_output = self.extend_cnn(inputs)
		# print(f'vgg.py 150 -- extended_output shape: {extended_output.shape}')
		### split extended output into 2 parts by channels
		#### csp_input.shape: [b, middle_channels,w,h]
		### residual.shape: [b, 512-middle_channels,w,h]
		csp_input, residual_input = torch.split(extended_output,self.middle_channels,dim=1)
		# print(f'vgg.py 150 -- csp_input shape: {csp_input.shape}')
		### pass csp_input goes throush vgg network
		vgg_output = self.vgg(csp_input)
		# print(f'vgg.py 150 -- vgg_output shape: {vgg_output.shape}')
		### decrease vgg output channels by a half using transition layer
		### transition_output.shape: [b, middle_channels, w,h]
		transition_output = self.transition_cnn(vgg_output)
		# print(f'vgg.py 150 -- transition_output shape: {transition_output.shape}')
		### concatnate residual input and transition_output (fusion last)
		### x.shape: [b,512,w,h]
		x = torch.cat((transition_output,residual_input),dim=1)
		# print(f'vgg.py 150 -- x shape: {x.shape}')

		x = self.max_pool(x)
		# print(f'vgg.py 150 -- x shape: {x.shape}')
		x = x.view(x.size(0), -1)
		out = self.classify(x)
		return out

	def cnn_layers(self,input_channel=3):
		layers = []
		#### vgg 19 architecture: 2 conv_64 + max pooling + 2 conv_128 + max_pooling + 4 conv_256 + max_pooling + (4 conv_512 + max_pooling) * 2

		for output_channel in cfg[self.vgg_type]:
			if output_channel == 'M':
				layers += [nn.MaxPool2d(kernel_size=1)]
				continue

			layers += [
				nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, padding=1)
			]

			if self.batch_norm:
				layers += [nn.BatchNorm2d(output_channel)]
			layers += [nn.ReLU(inplace=True)]
			input_channel = output_channel
		return nn.Sequential(*layers)








if __name__ == '__main__':
	csp_vgg = CSP_VGG(vgg_type='vgg19')
	vgg = VGG(vgg_type='vgg19')
	a = torch.arange(3 * 32 * 64 * 64).reshape(3, 32, 64, 64).float()
	max_pool = nn.MaxPool2d(kernel_size=8)
	b = max_pool(a)
	b.shape
	out = csp_vgg(a)
	# out = vgg(a)
	print(out.shape)
