# *******************************************************************
#
# Author : thien@locslab.com 2020
# with the reference of  : sthanhng@gmail.com
# on Github : https://github.com/sthanhng
# *******************************************************************
import colorsys
import os
import time

import torch

os.environ['KERAS_BACKEND'] = 'tensorflow'
import queue
import numpy as np
from PIL import ImageDraw, Image
import tensorflow as tf
config = tf.compat.v1.ConfigProto(device_count = {'GPU': 2} )
sess = tf.compat.v1.Session(config=config)
#import tensorflow.python.keras.backend as K  ### use for new tensorflow version
import tensorflow.compat.v1.keras.backend  as K
K.set_session(sess)
from models.common import Conv
from cv2 import cv2

tf.compat.v1.disable_eager_execution()
# from keras import backend as K ### use for old tensorflow version
#from tensorflow.python.keras.models import load_model
from tensorflow.keras.models import load_model
import threading
from modules.model import eval

gender_choice = {1: 'male', 2: 'female'}
age_choice = {1: '10대', 2: '20대', 3: '30대', 4: '40대', 5: '50대'}


# age_choice = {1: '<10', 2: '11~20', 3: '21~30', 4: '31~40', 5: '41~50', 6: '51~60', 7: '61~70', 8: '71~80', 9: '81~90', 10: '>90'}


def letterbox_image(image, size):
	'''Resize image with unchanged aspect ratio using padding'''

	img_width, img_height = image.size
	w, h = size
	scale = min(w / img_width, h / img_height)
	nw = int(img_width * scale)
	nh = int(img_height * scale)

	image = image.resize((nw, nh), Image.BICUBIC)
	new_image = Image.new('RGB', size, (128, 128, 128))
	new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
	return new_image


class YOLO(object):
	def __init__(self, args):
		self.args = args
		self.model_path = args.model
		self.classes_path = args.classes
		self.anchors_path = args.anchors
		self.class_names = self._get_class()
		self.anchors = self._get_anchors()
		self.sess = K.get_session()
		# self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
		self.boxes, self.scores, self.classes = self._generate()
		self.model_image_size = args.img_size

	def _get_class(self):
		classes_path = os.path.expanduser(self.classes_path)
		with open(classes_path) as f:
			class_names = f.readlines()
		class_names = [c.strip() for c in class_names]
		return class_names

	def _get_anchors(self):
		anchors_path = os.path.expanduser(self.anchors_path)
		with open(anchors_path) as f:
			anchors = f.readline()
		anchors = [float(x) for x in anchors.split(',')]
		return np.array(anchors).reshape(-1, 2)

	def _generate(self):
		model_path = os.path.expanduser(self.model_path)
		assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file'
		print(f'model_path: {self.model_path}')
		assert os.path.isfile(self.model_path)
		# load model, or construct model and load weights
		num_anchors = len(self.anchors)
		num_classes = len(self.class_names)
		try:
			self.yolo_model = load_model(model_path, compile=False)
		except:
			# make sure model, anchors and classes match
			self.yolo_model.load_weights(self.model_path)
		else:
			assert self.yolo_model.layers[-1].output_shape[-1] == \
			       num_anchors / len(self.yolo_model.output) * (
					       num_classes + 5), \
				'Mismatch between model and given anchor and class sizes'
		print(
			'*** {} model, anchors, and classes loaded.'.format(model_path))

		# generate colors for drawing bounding boxes
		hsv_tuples = [(x / len(self.class_names), 1., 1.)
		              for x in range(len(self.class_names))]
		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		self.colors = list(
			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
			    self.colors))

		# shuffle colors to decorrelate adjacent classes.
		np.random.shuffle(self.colors)
		# generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2,))
		boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
		                              len(self.class_names),
		                              self.input_image_shape,
		                              score_threshold=self.args.face_score,
		                              iou_threshold=self.args.iou)
		return boxes, scores, classes

	def detect_image(self, image):
		if self.model_image_size != (None, None):
			assert self.model_image_size[
				       0] % 32 == 0, 'Multiples of 32 required'
			assert self.model_image_size[
				       1] % 32 == 0, 'Multiples of 32 required'
			boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))

		else:
			new_image_size = (image.width - (image.width % 32),
			                  image.height - (image.height % 32))
			boxed_image = letterbox_image(image, new_image_size)

		image_data = np.array(boxed_image, dtype='float32')
		image_data /= 255.
		# add batch dimension
		image_data = np.expand_dims(image_data, 0)

		dict__ = {self.yolo_model.input: image_data, self.input_image_shape: [image.size[1], image.size[0]],
		          K.learning_phase(): 0}
		out_boxes, out_scores, out_classes = self.sess.run(
			[self.boxes, self.scores, self.classes],
			feed_dict={self.yolo_model.input: image_data,
			           self.input_image_shape: [image.size[1], image.size[0]],
			           # K.learning_phase(): 0
			           })
		thickness = (image.size[0] + image.size[1]) // 400

		for i, c in reversed(list(enumerate(out_classes))):
			box = out_boxes[i]
			draw = ImageDraw.Draw(image)
			top, left, bottom, right = box
			top = max(0, np.floor(top + 0.5).astype('int32'))
			left = max(0, np.floor(left + 0.5).astype('int32'))
			bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
			right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
			for thk in range(thickness):
				draw.rectangle([left + thk, top + thk, right - thk, bottom - thk], outline=(51, 178, 255))
			del draw
		return image, out_boxes

	def close_session(self):
		self.sess.close()


class Face(object):
	def __init__(self, id_, gender, age, first_centroid=None, last_centroid=None):
		self.id = id_
		self.gender = gender
		self.age = age
		self.first_centroid = first_centroid
		self.last_centroid = last_centroid


class Person(object):
	def __init__(self, id_, first_centroid=None, last_centroid=None):
		self.id = id_
		self.first_centroid = first_centroid
		self.last_centroid = last_centroid


# bufferless VideoCapture
class VideoCapture:
	def __init__(self, name, is_file=False):
		self.name = name
		self.cap = cv2.VideoCapture(self.name)

		self.q = queue.Queue()
		# self._reader()
		t = threading.Thread(target=self._reader)
		t.daemon = True
		t.start()

	# read frames as soon as they are available, keeping only most recent one
	def _reader(self):
		count = 0
		while True:
			ret, frame = self.cap.read()
			print('detecting..')
			if not ret:
				if count >= 1000:
					break
				else:
					count +=1
					del self.cap
					print('Camera error, recreating..!')
					self.cap = cv2.VideoCapture(self.name)
					continue
			count=0
			if not self.q.empty():
				try:
					self.q.get_nowait()  # discard previous (unprocessed) frame
				except queue.Empty:
					pass
			self.q.put(frame)

	def read(self):
		return self.q.get()

	def get_fps(self):
		return self.cap.get(cv2.CAP_PROP_FPS)

	def get_size(self):
		return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


class Ensemble(torch.nn.ModuleList):
	# Ensemble of models
	def __init__(self):
		super(Ensemble, self).__init__()

	def forward(self, x, augment=False):
		y = []
		for module in self:
			y.append(module(x, augment)[0])
		# y = torch.stack(y).max(0)[0]  # max ensemble
		# y = torch.stack(y).mean(0)  # mean ensemble
		y = torch.cat(y, 1)  # nms ensemble
		return y, None  # inference, train output


def attempt_load(weights, map_location=None):
	# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
	model = Ensemble()
	for w in weights if isinstance(weights, list) else [weights]:
		model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

	# Compatibility updates
	for m in model.modules():
		if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
			m.inplace = True  # pytorch 1.7.0 compatibility
		elif type(m) is Conv:
			m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

	if len(model) == 1:
		return model[-1]  # return model
	else:
		print('Ensemble created with %s\n' % weights)
		for k in ['names', 'stride']:
			setattr(model, k, getattr(model[-1], k))
		return model  # return ensemble


# # *******************************************************************
# #
# # Author : thien@locslab.com 2020
# # with the reference of  : sthanhng@gmail.com
# # on Github : https://github.com/sthanhng
# # *******************************************************************
# import colorsys
# import os
# import time
#
# import torch
#
# os.environ['KERAS_BACKEND'] = 'tensorflow'
# import queue
# import numpy as np
# from PIL import ImageDraw, Image
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 2} )
# sess = tf.compat.v1.Session(config=config)
# import tensorflow.python.keras.backend as K  ### use for new tensorflow version
# K.set_session(sess)
# from models.common import Conv
# from cv2 import cv2
#
# tf.compat.v1.disable_eager_execution()
# # from keras import backend as K ### use for old tensorflow version
# from keras.models import load_model
# import threading
# from modules.model import eval
#
# gender_choice = {1: 'male', 2: 'female'}
# age_choice = {1: '10대', 2: '20대', 3: '30대', 4: '40대', 5: '50대'}
#
#
# # age_choice = {1: '<10', 2: '11~20', 3: '21~30', 4: '31~40', 5: '41~50', 6: '51~60', 7: '61~70', 8: '71~80', 9: '81~90', 10: '>90'}
#
#
# def letterbox_image(image, size):
# 	'''Resize image with unchanged aspect ratio using padding'''
#
# 	img_width, img_height = image.size
# 	w, h = size
# 	scale = min(w / img_width, h / img_height)
# 	nw = int(img_width * scale)
# 	nh = int(img_height * scale)
#
# 	image = image.resize((nw, nh), Image.BICUBIC)
# 	new_image = Image.new('RGB', size, (128, 128, 128))
# 	new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
# 	return new_image
#
#
# class YOLO(object):
# 	def __init__(self, args):
# 		self.args = args
# 		self.model_path = args.model
# 		self.classes_path = args.classes
# 		self.anchors_path = args.anchors
# 		self.class_names = self._get_class()
# 		self.anchors = self._get_anchors()
# 		self.sess = K.get_session()
# 		# self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# 		self.boxes, self.scores, self.classes = self._generate()
# 		self.model_image_size = args.img_size
#
# 	def _get_class(self):
# 		classes_path = os.path.expanduser(self.classes_path)
# 		with open(classes_path) as f:
# 			class_names = f.readlines()
# 		class_names = [c.strip() for c in class_names]
# 		return class_names
#
# 	def _get_anchors(self):
# 		anchors_path = os.path.expanduser(self.anchors_path)
# 		with open(anchors_path) as f:
# 			anchors = f.readline()
# 		anchors = [float(x) for x in anchors.split(',')]
# 		return np.array(anchors).reshape(-1, 2)
#
# 	def _generate(self):
# 		model_path = os.path.expanduser(self.model_path)
# 		assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file'
# 		print(f'model_path: {self.model_path}')
# 		assert os.path.isfile(self.model_path)
# 		# load model, or construct model and load weights
# 		num_anchors = len(self.anchors)
# 		num_classes = len(self.class_names)
# 		try:
# 			self.yolo_model = load_model(model_path, compile=False)
# 		except:
# 			# make sure model, anchors and classes match
# 			self.yolo_model.load_weights(self.model_path)
# 		else:
# 			assert self.yolo_model.layers[-1].output_shape[-1] == \
# 			       num_anchors / len(self.yolo_model.output) * (
# 					       num_classes + 5), \
# 				'Mismatch between model and given anchor and class sizes'
# 		print(
# 			'*** {} model, anchors, and classes loaded.'.format(model_path))
#
# 		# generate colors for drawing bounding boxes
# 		hsv_tuples = [(x / len(self.class_names), 1., 1.)
# 		              for x in range(len(self.class_names))]
# 		self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
# 		self.colors = list(
# 			map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
# 			    self.colors))
#
# 		# shuffle colors to decorrelate adjacent classes.
# 		np.random.shuffle(self.colors)
# 		# generate output tensor targets for filtered bounding boxes.
# 		self.input_image_shape = K.placeholder(shape=(2,))
# 		boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
# 		                              len(self.class_names),
# 		                              self.input_image_shape,
# 		                              score_threshold=self.args.face_score,
# 		                              iou_threshold=self.args.iou)
# 		return boxes, scores, classes
#
# 	def detect_image(self, image):
# 		if self.model_image_size != (None, None):
# 			assert self.model_image_size[
# 				       0] % 32 == 0, 'Multiples of 32 required'
# 			assert self.model_image_size[
# 				       1] % 32 == 0, 'Multiples of 32 required'
# 			boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
#
# 		else:
# 			new_image_size = (image.width - (image.width % 32),
# 			                  image.height - (image.height % 32))
# 			boxed_image = letterbox_image(image, new_image_size)
#
# 		image_data = np.array(boxed_image, dtype='float32')
# 		image_data /= 255.
# 		# add batch dimension
# 		image_data = np.expand_dims(image_data, 0)
#
# 		dict__ = {self.yolo_model.input: image_data, self.input_image_shape: [image.size[1], image.size[0]],
# 		          K.learning_phase(): 0}
# 		out_boxes, out_scores, out_classes = self.sess.run(
# 			[self.boxes, self.scores, self.classes],
# 			feed_dict={self.yolo_model.input: image_data,
# 			           self.input_image_shape: [image.size[1], image.size[0]],
# 			           # K.learning_phase(): 0
# 			           })
# 		thickness = (image.size[0] + image.size[1]) // 400
#
# 		for i, c in reversed(list(enumerate(out_classes))):
# 			box = out_boxes[i]
# 			draw = ImageDraw.Draw(image)
# 			top, left, bottom, right = box
# 			top = max(0, np.floor(top + 0.5).astype('int32'))
# 			left = max(0, np.floor(left + 0.5).astype('int32'))
# 			bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
# 			right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
# 			for thk in range(thickness):
# 				draw.rectangle([left + thk, top + thk, right - thk, bottom - thk], outline=(51, 178, 255))
# 			del draw
# 		return image, out_boxes
#
# 	def close_session(self):
# 		self.sess.close()
#
#
# class Face(object):
# 	def __init__(self, id_, gender, age, first_centroid=None, last_centroid=None):
# 		self.id = id_
# 		self.gender = gender
# 		self.age = age
# 		self.first_centroid = first_centroid
# 		self.last_centroid = last_centroid
#
#
# class Person(object):
# 	def __init__(self, id_, first_centroid=None, last_centroid=None):
# 		self.id = id_
# 		self.first_centroid = first_centroid
# 		self.last_centroid = last_centroid
#
#
# # bufferless VideoCapture
# class VideoCapture:
# 	def __init__(self, name):
# 		self.cap = cv2.VideoCapture(name)
# 		self.q = queue.Queue()
# 		t = threading.Thread(target=self._reader)
# 		t.daemon = True
# 		t.start()
#
# 	# read frames as soon as they are available, keeping only most recent one
# 	def _reader(self):
# 		count = 0
# 		while True:
# 			ret, frame = self.cap.read()
#
# 			if not ret:
# 				if count>=1000:
# 					break
# 				else:
# 					count +=1
# 					continue
# 			count=0
# 			if not self.q.empty():
# 				try:
# 					self.q.get_nowait()  # discard previous (unprocessed) frame
# 				except queue.Empty:
# 					pass
# 			self.q.put(frame)
#
# 	def read(self):
# 		return self.q.get()
#
# 	def get_fps(self):
# 		return self.cap.get(cv2.CAP_PROP_FPS)
#
# 	def get_size(self):
# 		return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#
# class Ensemble(torch.nn.ModuleList):
# 	# Ensemble of models
# 	def __init__(self):
# 		super(Ensemble, self).__init__()
#
# 	def forward(self, x, augment=False):
# 		y = []
# 		for module in self:
# 			y.append(module(x, augment)[0])
# 		# y = torch.stack(y).max(0)[0]  # max ensemble
# 		# y = torch.stack(y).mean(0)  # mean ensemble
# 		y = torch.cat(y, 1)  # nms ensemble
# 		return y, None  # inference, train output
#
#
# def attempt_load(weights, map_location=None):
# 	# Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
# 	model = Ensemble()
# 	for w in weights if isinstance(weights, list) else [weights]:
# 		model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model
#
# 	# Compatibility updates
# 	for m in model.modules():
# 		if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
# 			m.inplace = True  # pytorch 1.7.0 compatibility
# 		elif type(m) is Conv:
# 			m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
#
# 	if len(model) == 1:
# 		return model[-1]  # return model
# 	else:
# 		print('Ensemble created with %s\n' % weights)
# 		for k in ['names', 'stride']:
# 			setattr(model, k, getattr(model[-1], k))
# 		return model  # return ensemble
