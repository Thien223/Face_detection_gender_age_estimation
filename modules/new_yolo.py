# *******************************************************************
#
# Author : thien@locslab.com 2020
# with the reference of  : sthanhng@gmail.com
# on Github : https://github.com/sthanhng
# *******************************************************************
import colorsys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import ImageDraw, Image
from keras import backend as K
from keras.models import load_model

from modules.model import eval
from utils.trac_object import CentroidTracker

sex_choice = {1: 'male', 2: 'female'}
age_choice = {1: '<26', 2: '27~37', 3: '37~48', 4: '49~59', 5: '>60'}
# age_choice = {1: '<10', 2: '11~20', 3: '21~30', 4: '31~40', 5: '41~50', 6: '51~60', 7: '61~70', 8: '71~80', 9: '81~90', 10: '>90'}


class YOLO(object):
	def __init__(self, args):
		self.args = args
		self.model_path = args.model
		self.classes_path = args.classes
		self.anchors_path = args.anchors
		self.class_names = self._get_class()
		self.anchors = self._get_anchors()
		self.sess = K.get_session()
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
		assert model_path.endswith(
			'.h5'), 'Keras model or weights must be a .h5 file'

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
		np.random.seed(102)
		np.random.shuffle(self.colors)
		np.random.seed(None)

		# generate output tensor targets for filtered bounding boxes.
		self.input_image_shape = K.placeholder(shape=(2,))
		boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
										   len(self.class_names),
										   self.input_image_shape,
										   score_threshold=self.args.score,
										   iou_threshold=self.args.iou)
		return boxes, scores, classes

	def detect_image(self, image):
		if self.model_image_size != (None, None):
			assert self.model_image_size[
					   0] % 32 == 0, 'Multiples of 32 required'
			assert self.model_image_size[
					   1] % 32 == 0, 'Multiples of 32 required'
			boxed_image = letterbox_image(image, tuple(
				reversed(self.model_image_size)))
		else:
			new_image_size = (image.width - (image.width % 32),
							  image.height - (image.height % 32))
			boxed_image = letterbox_image(image, new_image_size)
		image_data = np.array(boxed_image, dtype='float32')
		image_data /= 255.
		# add batch dimension
		image_data = np.expand_dims(image_data, 0)
		out_boxes, out_scores, out_classes = self.sess.run(
			[self.boxes, self.scores, self.classes],
			feed_dict={
				self.yolo_model.input: image_data,
				self.input_image_shape: [image.size[1], image.size[0]],
				K.learning_phase(): 0
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
				draw.rectangle(
					[left + thk, top + thk, right - thk, bottom - thk],
					outline=(51, 178, 255))
			del draw
		return image, out_boxes

	def close_session(self):
		self.sess.close()


class Face(object):
	def __init__(self, id, gender, age, first_centroid=None, last_centroid=None):
		self.id = id
		self.gender=gender
		self.age=age
		self.first_centroid=first_centroid
		self.last_centroid=last_centroid


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


def detect_img(model, image_path, age_gender_model=None):
	max_= 20
	f_count=0
	m_count = 0
	male_count = {}
	female_count = {}
	male_true=0
	male_false=0
	female_true=0
	female_false=0

	age_true={}
	age_false={}
		# temp = '00866A15'
	for file in os.listdir(image_path):
		full_path = os.path.join(image_path, file)
		print(f'male_count {m_count} -- female_count {f_count}')
		print(full_path)
		if not file.endswith('.jpg'):
			continue
		temp = file.replace('.jpg','').replace('-with-mask','')
		prefix, age = temp.split('A')
		# _, age, gender, filename = full_path.split('\\')
		age = int(age)
		# gender = "male" if gender == "111" else "female"
		gender = 0 if int(prefix) >= 8147 else 1


		if age < 26:
			age_ = 0
		elif age > 26 and age <= 37:
			age_ = 1
		elif age > 37 and age <= 48:
			age_ = 2
		elif age > 48 and age <= 59:
			age_ = 3
		else:
			age_ = 4

		if gender==0:
			if f_count>=m_count:
				m_count+=1
				if age in male_count.keys():
					if male_count[age] < max_:
						try:
							pred_gender, pred_age = detect(model=model, file_path=full_path, age_gender_model=age_gender_model)
							pred_gender, pred_age = int(pred_gender), int(pred_age)
							if pred_gender == gender:
								male_true += 1
							elif pred_gender is None:
								continue
							else:
								male_false += 1


							if pred_age==age_:
								if age_ in age_true.keys():
									age_true[age_] +=1
								else:
									age_true[age_] = 1
							elif pred_age is None:
								continue
							else:

								if age_ in age_false.keys():
									if pred_age in age_false[age_].keys():
										age_false[age_][pred_age] += 1
									else:
										age_false[age_][pred_age] = 1
								else:
									age_false[age_] = {}
									age_false[age_][pred_age] = 1

							male_count[age] += 1
						except:
							continue
					else:
						continue
				else:
					try:
						pred_gender, pred_age = detect(model=model, file_path=full_path, age_gender_model=age_gender_model)
						pred_gender, pred_age = int(pred_gender), int(pred_age)
						if pred_gender == gender:
							male_true += 1
						elif pred_gender is None:
							continue
						else:
							male_false += 1
						if pred_age == age_:
							if age_ in age_true.keys():
								age_true[age_] += 1
							else:
								age_true[age_] = 1
						elif pred_age is None:
							continue
						else:
							if age_ in age_false.keys():
								if pred_age in age_false[age_].keys():
									age_false[age_][pred_age] += 1
								else:
									age_false[age_][pred_age] = 1
							else:
								age_false[age_] = {}
								age_false[age_][pred_age] = 1
						male_count[age] = 1
					except:
						continue
		else:
			f_count += 1
			if age in female_count.keys():
				if female_count[age] < max_:
					try:
						pred_gender, pred_age = detect(model=model, file_path=full_path, age_gender_model=age_gender_model)
						pred_gender, pred_age = int(pred_gender), int(pred_age)
						if pred_gender == gender:
							female_true += 1
						elif pred_gender is None:
							continue
						else:
							female_false += 1
						if pred_age == age_:
							if age_ in age_true.keys():
								age_true[age_] += 1
							else:
								age_true[age_] = 1
						elif pred_age is None:
							continue
						else:
							if age_ in age_false.keys():
								if pred_age in age_false[age_].keys():
									age_false[age_][pred_age] += 1
								else:
									age_false[age_][pred_age] = 1
							else:
								age_false[age_] = {}
								age_false[age_][pred_age] = 1
						female_count[age] += 1
					except:
						continue
				else:
					continue
			else:
				try:
					pred_gender, pred_age = detect(model=model, file_path=full_path, age_gender_model=age_gender_model)
					pred_gender, pred_age = int(pred_gender), int(pred_age)
					if pred_gender == gender:
						female_true += 1
					elif pred_gender is None:
						continue
					else:
						female_false += 1
					if pred_age == age_:
						if age_ in age_true.keys():
							age_true[age_] += 1
						else:
							age_true[age_] = 1
					elif pred_age is None:
						continue
					else:
						if age_ in age_false.keys():
							if pred_age in age_false[age_].keys():
								age_false[age_][pred_age] += 1
							else:
								age_false[age_][pred_age] = 1
						else:
							age_false[age_] = {}
							age_false[age_][pred_age] = 1
					female_count[age] = 1
				except:
					continue

	print(f'male_true {male_true}')
	print(f'male_false {male_false}')
	print(f'female_true {female_true}')
	print(f'female_false {female_false}\n\n')

	print(f'age_true {age_true}')
	print(f'age_false {age_false}\n\n')
	print(f'male_count {male_count}')
	print(f'female_count {female_count}')
	model.close_session()



def detect(model, file_path, age_gender_model):
	image = Image.open(file_path)
	image, faces = model.detect_image(image)
	image = np.asarray(image)
	gender = None
	age = None
	for i, (x1, y1, x2, y2) in enumerate(faces):
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		face_img = image[x1: x2, y1:y2].copy()
		# Predict Gender and Age
		try:
			#### preparing face vector before passing to age, gender estimating model
			vectors = image_loader_(face_img)
			age_gender_output = age_gender_model(vectors)
			## convert predicted index to meaningful label
			# sex_indicate = [i + 1 for i in age_gender_output['sex'].argmax(dim=-1).tolist()]
			# age_indicate = [i + 1 for i in age_gender_output['age'].argmax(dim=-1).tolist()]
			gender = age_gender_output['sex'].argmax(dim=-1)
			age = age_gender_output['age'].argmax(dim=-1)
		except Exception as e:
			print(e)
			continue

	return gender, age



def image_loader_(face_img):
	from PIL import Image
	from torchvision import transforms
	import numpy as np
	img_size = 64
	image_set = []

	preprocess = transforms.Compose([transforms.Resize(img_size),
									 transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],
														  std=[0.229, 0.224, 0.225])])
	face_img = Image.fromarray(np.uint8(face_img))
	img = preprocess(face_img)
	image_set.append(img)
	img = torch.stack(image_set, dim=0)
	return img


def detect_video(model, video_path=None, age_gender_model=None):
	if video_path == 'stream':
		vid = cv2.VideoCapture(0)
	else:
		vid = cv2.VideoCapture(video_path)
	if not vid.isOpened():
		raise IOError("Couldn't open webcam or video")
	# the video format and fps
	# video_fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
	tracker = CentroidTracker()
	saved_object_ids = []
	face_objs=[]
	save_output=True
	ret=True
	out_stream_writer=None
	out_video_filename = video_path.split('/')[-1].split('.')[0]
	if save_output:
		video_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		video_fps = vid.get(cv2.CAP_PROP_FPS)
		# the size of the frames to write
		out_stream_writer = cv2.VideoWriter(f'outputs/{out_video_filename}.avi', video_fourcc, video_fps, (int(vid.get(3)), int(vid.get(4))))
	try:
		while ret:
			current_object_ids = []
			ret, frames = vid.read()
			try:
				image = Image.fromarray(frames)
			except AttributeError as e:
				print(f' new_yolo.py line 414: {e}')
				continue
			image, faces = model.detect_image(image)
			gender = 'unknown'
			age = 'unknown'
			tracking_faces = []

			for i, (x1, y1, x2, y2) in enumerate(faces): ## with yolo, result will be 2 point of rectangle corner (x1, y1) and (x2, y2)
				### each time detected a face, insert a new color
				(x1, y1, x2, y2)=(int(x1), int(y1), int(x2), int(y2))
				tracking_faces.append([y1, x1, y2, x2])
				### extract the face
				face_img = frames[x1: x2, y1:y2].copy()
				cv2.rectangle(frames, (y1, x1), (y2, x2), (255, 0, 0), 2)
				# Predict Gender and Age
				try:
					#### preparing face vector before passing to age, gender estimating model
					vectors = image_loader_(face_img)
					age_gender_output = age_gender_model(vectors)
					## convert predicted index to meaningful label
					sex_indicate = [i + 1 for i in age_gender_output['sex'].argmax(dim=-1).tolist()]
					age_indicate = [i + 1 for i in age_gender_output['age'].argmax(dim=-1).tolist()]
					gender = sex_choice.get(sex_indicate[0])
					age = age_choice.get(age_indicate[0])
				except Exception as e:
					continue
			objects = tracker.update(tracking_faces)
			print(f'Saved object ids: {saved_object_ids}')
			for (object_id, centroid) in objects.items():
				current_object_ids.append(object_id)
				if object_id not in saved_object_ids:
					if (gender!='unknown' and age!='unknown'):
						print(f'there are new object: {object_id}')
						## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
						new_face = Face(id=object_id,gender=[gender],age=[age],first_centroid=centroid)
						face_objs.append(new_face)
						saved_object_ids.append(object_id)

				else:
					if (gender != 'unknown' and age != 'unknown'):
						# print(f'object_id {object_id}')
						# print(f'saved_object_ids {saved_object_ids}')
						### when the face object is already in the managing face_objects, update it's info
						### get and edit
						old_face = face_objs[saved_object_ids.index(object_id)]
						old_face.gender = old_face.gender + [gender]
						old_face.age = old_face.age + [age]
						old_face.last_centroid=centroid
						### update
						face_objs[saved_object_ids.index(object_id)] = old_face
				#### draw rectangle bounding box for each face
				text = "ID {}, gender {}, age {}".format(object_id, gender, age)
				cv2.putText(frames, text, (centroid[0] - 10, centroid[1] - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frames, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
			# print(f'len(face_objs): {len(face_objs)}')
			# print(f'current_object_ids: {current_object_ids}')
			for obj in face_objs:
				if obj.id not in current_object_ids: ### face disappeared
					gender = 'Male' if (obj.gender.count('male') >= obj.gender.count('female')) else 'Female'
					age = max(set(obj.age), key = obj.age.count)
					try:
						going_in = True if obj.first_centroid[-1] > obj.last_centroid[-1] else False
						### remove disappeared object from face_objs and saved face_id
						face_objs.remove(obj)
						saved_object_ids.remove(obj.id)
						print(f'id: {obj.id}')
						print(f'gender: {gender}')
						print(f'age: {age}')
						print(f'going_in: {going_in}')
						# send(obj.id, gender, age, going_in)
					except AttributeError as e:
						face_objs.remove(obj)
						saved_object_ids.remove(obj.id)
						continue

			# cv2.imshow("Map View", frames)

			### save video if needed
			if save_output:
				out_stream_writer.write(frames)

			#### define interupt event
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	except KeyboardInterrupt:
		print('keyboard interupted..!')
	finally:
		vid.release()
		out_stream_writer.release()
		cv2.destroyAllWindows()
		# close the session
		model.close_session()



def send(id, gender, age, going_in):
	import requests
	import json
	URL = 'http://211.236.48.214:8000'
	headers = {
		'Content-Type': 'application/json',
	}
	str_ = 'mutation{createDetection(identifier:%d, age:\"%s\", gender:\"%s\", inAndOut:%s){identifier age gender inAndOut}}' % (id, age, gender, str(going_in).lower())
	data = dict(
		query=str_,
		variables={}
	)
	print(data)
	requests.post(URL, data=json.dumps(data), headers=headers)