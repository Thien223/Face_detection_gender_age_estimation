### thien.locslab@gmail.com
import argparse
import os
import random
import time
from pathlib import Path
import threading
import numpy as np
# import tensorflow as tf
import torch
from PIL import Image
from cv2 import cv2

from modules.new_yolo import Face, Person, YOLO, attempt_load, VideoCapture
from train_module.data_helper import _age_categorization
from utils.datasets import LoadStreams, letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, \
	xyxy2xywh, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized
from utils.trac_object import CentroidTracker

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
np.random.seed(1542)
# tf.random.set_seed(1542)
# tf.compat.v1.disable_eager_execution()

os.environ['KERAS_BACKEND'] = 'tensorflow'

gender_choice = {0: 'female', 1: 'male'}
age_choice = {0: '10', 1: '20', 2: '30', 3: '40', 4: '50'}


# age_choice = {1: '<10', 2: '11~20', 3: '21~30', 4: '31~40', 5: '41~50', 6: '51~60', 7: '61~70', 8: '71~80', 9: '81~90', 10: '>90'}


def detect_img(model, image_path, age_gender_model=None):
	max_ = 20000
	f_count = 0
	m_count = 0
	male_count = {}
	female_count = {}
	male_true = 0
	male_false = 0
	female_true = 0
	female_false = 0

	age_true = {}
	age_false = {}

	# temp = '00866A15'

	### for tarbal test set
	################################
	# image_path = 'dataset/train_image_unbalanced'
	# with open(os.path.join(image_path, 'metadata.json'), 'r') as tarbal:
	#     paths = json.load(tarbal)
	# random.shuffle(paths)
	# for dict_ in paths:
	#     full_path = dict_['image']
	#     age = dict_['age']
	#     gender = 1 if dict_['gender'] == 'male' else 0
	#     if m_count >= 4500: break
	####################################################

	####################################################
	# for all_face dataset
	files = os.listdir(image_path)
	random.shuffle(files)
	for file in files:
		full_path = os.path.join(image_path, file)
		# print(full_path)
		if not (file.endswith('.png') or file.endswith('.jpg')):
			continue
		temp = file.replace('.png', '').replace('.jpg', '').replace('-with-mask', '')
		# prefix, age = temp.split('_')[-2:]
		prefix, age = temp.split('A')
		# _, age, gender, filename = full_path.split('\\')
		# gender = "male" if gender == "111" else "female"
		gender = 1 if int(prefix) >= 8147 else 0  ### 1 is male, 0 is female
		####################################################
		# gender = int(prefix)
		age = int(age)
		age_ = _age_categorization(age)

		if gender == 1:
			if f_count >= m_count:
				m_count += 1
				if age in male_count.keys():
					if male_count[age] < max_:
						try:
							pred_gender, pred_age = detect(model=model, file_path=full_path,
														   age_gender_model=age_gender_model)
							pred_gender, pred_age = round(float(pred_gender)), round(float(pred_age))
							print(f'pred_age {pred_age} -- real: {age_}')
							if pred_gender == gender:
								male_true += 1
							elif pred_gender is None:
								continue
							else:
								male_false += 1
							if (age_ - 0.5) < pred_age <= age_ + 0.5:
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

							male_count[age] += 1
						except Exception as e:
							print(f'Run_yolo 201 error: {e}')
							continue
					else:
						continue
				else:
					try:
						pred_gender, pred_age = detect(model=model, file_path=full_path,
													   age_gender_model=age_gender_model)
						pred_gender, pred_age = round(float(pred_gender)), round(float(pred_age))
						print(f'pred_age {pred_age} -- real: {age_}')
						if pred_gender == gender:
							male_true += 1
						elif pred_gender is None:
							continue
						else:
							male_false += 1
						if (age_ - 0.5) < pred_age <= age_ + 0.5:
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
					except Exception as e:
						print(f'Run_yolo 235 error: {e}')
						continue
		else:
			f_count += 1
			if age in female_count.keys():
				if female_count[age] < max_:
					try:
						pred_gender, pred_age = detect(model=model, file_path=full_path,
													   age_gender_model=age_gender_model)
						pred_gender, pred_age = round(float(pred_gender)), round(float(pred_age))
						print(f'pred_age {pred_age} -- real: {age_}')
						if pred_gender == gender:
							female_true += 1
						elif pred_gender is None:
							continue
						else:
							female_false += 1
						if (age_ - 0.5) < pred_age <= age_ + 0.5:
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
					except Exception as e:
						print(f'Run_yolo 273 error: {e}')
						continue
				else:
					continue
			else:
				try:
					pred_gender, pred_age = detect(model=model, file_path=full_path, age_gender_model=age_gender_model)
					pred_gender, pred_age = round(float(pred_gender)), round(float(pred_age))
					print(f'pred_age {pred_age} -- real: {age_}')
					if pred_gender == gender:
						female_true += 1
					elif pred_gender is None:
						continue
					else:
						female_false += 1
					if (age_ - 0.5) < pred_age <= age_ + 0.5:
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
				except Exception as e:
					print(f'Run_yolo 306 error: {e}')
					continue
	print(f'male_true {male_true}')
	print(f'male_false {male_false}')
	print(f'female_true {female_true}')
	print(f'female_false {female_false}\n\n')
	age_true_count = 0
	age_false_count = 0
	for key in age_true.keys():
		age_true_count += int(age_true[key])
	for _age in [0, 1, 2, 3, 4]:
		if _age in age_false.keys():
			for key in age_false[_age].keys():
				age_false_count += int(age_false[_age][key])
	print(f'age_true {age_true}')
	print(f'age_false {age_false}\n\n')
	print(f'age_true_count {age_true_count}')
	print(f'age_false_count {age_false_count}\n\n')
	print(f'male_count {male_count}')
	print(f'female_count {female_count}\n\n')
	model.close_session()


def image_loader_(face_img):
	from PIL import Image
	from torchvision import transforms
	import numpy as np
	from train_module.data_helper import img_size  #### img_size=64
	# image_set = []
	preprocess = transforms.Compose([transforms.Resize(img_size + 4),
									 transforms.RandomCrop(img_size),
									 transforms.RandomHorizontalFlip(),
									 transforms.ToTensor(),
									 # transforms.Normalize(mean=mean,
									 #                      std=std)
									 ])
	# face_img = Image.fromarray(np.uint8(face_img))
	face_img = Image.fromarray(np.uint8(face_img))
	# face_img = Image.open(face_img)
	img = preprocess(face_img)
	img = img.unsqueeze(dim=0)
	return img


def image_loader(face_img_path):
	from PIL import Image
	from train_module.data_helper import image_preprocess
	# image_set = []
	preprocess = image_preprocess(img_path=None)
	face_img = Image.open(face_img_path)
	img = preprocess(face_img)
	img = img.unsqueeze(dim=0)
	return img


def send(id, gender, age, going_in):
	import requests
	import json
	URL = 'http://museum.locslab.com/api'
	headers = {
		'Content-Type': 'application/json',
	}
	str_ = 'mutation{createDetection(identifier:%d, age:\"%s\", gender:\"%s\", inAndOut:%s){identifier age gender inAndOut}}' % (
		id, age, gender, str(going_in).lower())
	data = dict(
		query=str_,
		variables={}
	)
	print(data)
	requests.post(URL, data=json.dumps(data), headers=headers)


### get pretrained model of age and gender detection
def get_model(model_path=None):
	if model_path is None:
		model_path = 'models/checkpoints/vgg-epochs_88-step_0-gender_acc_98.24347172647003-age_acc_78.55845701045776.pth'
	# model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
	checkpoint = torch.load(model_path, map_location=device)
	model_type = checkpoint['model_type']
	if model_type == 'vgg':
		from modules.vgg import Gender_VGG, Age_VGG
		gender_model = Gender_VGG(vgg_type='vgg19')
		age_model = Age_VGG(vgg_type='vgg19')
		gender_model.load_state_dict(checkpoint['gender_model_weights'])
		age_model.load_state_dict(checkpoint['age_model_weights'])
		gender_model.eval()
		age_model.eval()
		return gender_model, age_model
	elif model_type == 'cspvgg':
		from modules.vgg import CSP_VGG
		m = CSP_VGG(vgg_type='vgg19')
		m.load_state_dict(checkpoint['model_state_dict'])
		m.eval()
	elif model_type == 'inception':
		from modules.vgg import Inception_VGG
		m = Inception_VGG(vgg_type='vgg19')
		m.load_state_dict(checkpoint['model_state_dict'])
		m.eval()
		return m


def detect_out(model, args):
	if args.half:
		model.half()
	"""
	people recognition to detect going out people
	"""
	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	video_path = args.video
	# Directories
	stride = int(model.stride.max())  # model stride
	img_size = check_img_size(640, s=stride)  # check img_size
	# Set Dataloader
	vid_path, vid_writer = None, None
	#### tracking objects
	tracker = CentroidTracker(maxDisappeared=60)
	### detected object ids
	saved_object_ids = []
	### person objects list
	person_objs = []
	#
	# dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
	# for path, img, im0s, vid_cap in dataset:
	# 	img = torch.from_numpy(img).to(device)
	# 	img = img.half() if args.half else img.float()  # uint8 to fp16/32
	# 	img /= 255.0  # 0 - 255 to 0.0 - 1.0
	# 	if img.ndimension() == 3:
	# 		img = img.unsqueeze(0)
	# 	# Inference
	# 	t1 = time_synchronized()
	cap = VideoCapture(video_path)
	while True:
		try:
			frames = cap.read()
			# ## for testing: set frames to None and check what will happen
			# if int(time.time()) % 99 == 0:
			# 	print('setting frames to None')
			# 	frames = None
			img = [letterbox(x, img_size, auto=True, stride=stride)[0] for x in [frames]]
			# Stack
			img = np.stack(img, 0)
			# Convert
			img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
			img = np.ascontiguousarray(img)
			img = torch.from_numpy(img).to(device)
			img = img.half() if args.half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)
			with torch.no_grad():
				pred = model(img, augment=False)[0]
			# Apply NMS
			pred = non_max_suppression(pred, args.person_score, args.iou, classes=0, agnostic=False)
			tracking_obj = []
			# # Process detections
			for i, det in enumerate(pred):  # detections per image
				# 	# Write resultsqq
				for *xyxy, conf, cls in reversed(det):
					if names[int(cls)] == 'person':
						#### tracking people in curent frame
						(x1, y1, x2, y2) = (int(xyxy[0]) * 2, int(xyxy[1]) * 2, int(xyxy[2]) * 2, int(xyxy[3]) * 2)
						tracking_obj.append([x1, y1, x2, y2])
						if args.save_img or args.view_img:  # Add bbox to image
							label = f'{names[int(cls)]} {conf:.2f}'
							cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 255, 0), 2)
				### update tracking objects
				objects = tracker.update(tracking_obj)
				### object ids in current frame (reset each frame)
				current_object_ids = set()
				for (object_id, centroid) in objects.items():
					current_object_ids.add(object_id)
					if object_id not in saved_object_ids:
						## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
						new_person = Person(id_=object_id, first_centroid=centroid)
						person_objs.append(new_person)
						saved_object_ids.append(object_id)
					else:
						### when the face object is already in the managing face_objects, update it's info
						### get and edit
						old_person = person_objs[saved_object_ids.index(object_id)]
						old_person.last_centroid = centroid
						### update
						person_objs[saved_object_ids.index(object_id)] = old_person
					#### draw rectangle bounding box for each face
					text = f"ID:{object_id}"
					# print(f'\n===============================')
					# print(text)
					# print(f'===============================\n')
					cv2.putText(img=frames,
								text=text,
								org=(centroid[0] - 10, centroid[1] - 10),
								fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								fontScale=0.55,
								color=(0, 255, 0),
								thickness=2)
					cv2.circle(img=frames,
							   center=(centroid[0], centroid[1]),
							   radius=4,
							   color=(0, 255, 0),
							   thickness=1)
				for obj in person_objs:
					if obj.id not in current_object_ids:  ### face disappeared
						### human recognition model does not have gender and age info
						gender = 'unknown'
						age = -1
						try:
							going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
							### remove disappeared object from face_objs and saved face_id
							person_objs.remove(obj)
							saved_object_ids.remove(obj.id)
							# print(f'id: {obj.id}')
							# print(f'gender: {gender}')
							# print(f'age: {age}')
							# print(f'going_in: {going_in}')
							# txt = f'id: {obj.id}\ngender: {gender}\nage: {age}\ngoing_in: {going_in}\n'
							# yield (f'<br><br><br>id: {obj.id}<br>gender: {gender}<br>age: {age}<br>going_in: {going_in}')
							if not going_in:
								print(f'Someone is going out')
								send(obj.id, gender, age, going_in)
						except Exception as e:
							person_objs.remove(obj)
							saved_object_ids.remove(obj.id)
							continue
				# Print time (inference + NMS)
				# print(f'{s}Done. ({t2 - t1:.3f}s)')
				# Stream results
				if args.view_img:
					cv2.imshow('detecting out', frames)
			# Save results (image with detections)
			# if args.save_img:
			# 	if dataset.mode == 'image':
			# 		cv2.imwrite(save_path, im0)
			# 	else:  # 'video'
			# 		if vid_path != save_path:  # new video
			# 			vid_path = save_path
			# 			if isinstance(vid_writer, cv2.VideoWriter):
			# 				vid_writer.release()  # release previous video writer
			#
			# 			fourcc = 'mp4v'  # output video codec
			# 			fps = vid_cap.get(cv2.CAP_PROP_FPS)
			# 			w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			# 			h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			# 			vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
			# 		vid_writer.write(im0)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		except Exception as e:
			print(f'run_yolo.py 547. Error: {e}')
			cap = VideoCapture(video_path)
			print(f'Recreated video capture object')
			continue
	cv2.destroyAllWindows()


def detect_in(model, age_gender_model, args):
	"""
	face detection to detect going in people
	"""
	# the video format and fps
	# video_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

	assert isinstance(age_gender_model, tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
	gender_model, age_model = age_gender_model
	if args.half:
		model.half()
	"""
	people recognition to detect going out people
	"""
	# Get names and colors
	video_path = args.video
	# Set Dataloader
	tracker = CentroidTracker(maxDisappeared=60)
	saved_object_ids = []
	face_objs = []
	# dataset = LoadStreams(video_path)
	cap = VideoCapture(video_path)
	expand_ratio = 0.0

	while True:
		try:
			frames = cap.read()
			image = Image.fromarray(frames)
			with torch.no_grad():
				image, faces = model.detect_image(image)
			tracking_faces = []

			for i, (x1, y1, x2, y2) in enumerate(faces):  ## with yolo, result will be 2 point of rectangle corner (x1, y1) and (x2, y2)
				try:
					(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
					tracking_faces.append([y1, x1, y2, x2])

					x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
					x2 = min(x2 + (expand_ratio * (x2 - x1)), frames.shape[0])
					y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
					y2 = min(y2 + (expand_ratio * (y2 - y1)), frames.shape[1])
					(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))

					### extract the face
					face_img = frames[x1: x2, y1:y2].copy()
					cv2.rectangle(frames, (y1, x1), (y2, x2), (255, 0, 0), 2)
					# Predict Gender and Age
					with torch.no_grad():
						#### preparing face vector before passing to age, gender estimating model
						vectors = image_loader_(face_img)
						# age_gender_output = age_gender_model(vectors)
						pred_gender = gender_model(vectors)
						pred_age = age_model(vectors)
						## convert predicted index to meaningful label
						gender_indicate = pred_gender.argmax(dim=1).item()
						age_indicate = round(float(pred_age))
						gender = gender_choice.get(gender_indicate)
						age = age_choice.get(age_indicate)
						print(f'detected! gender: {gender}')
				except Exception as e:
					print(f'run_yolo_new.py line 510. Error {e}')
					continue
			objects = tracker.update(tracking_faces)
			current_object_ids = set()
			for (object_id, centroid) in objects.items():
				current_object_ids.add(object_id)
				if object_id not in saved_object_ids:
					if gender != 'unknown' and age != 'unknown':
						# print(f'there are new object: {object_id}')
						## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
						new_face = Face(id_=object_id, gender=[gender], age=[age], first_centroid=centroid)
						face_objs.append(new_face)
						saved_object_ids.append(object_id)

				else:
					if gender != 'unknown' and age != 'unknown':
						# print(f'object_id {object_id}')
						# print(f'saved_object_ids {saved_object_ids}')
						### when the face object is already in the managing face_objects, update it's info
						### get and edit
						old_face = face_objs[saved_object_ids.index(object_id)]
						old_face.gender = old_face.gender + [gender]
						old_face.age = old_face.age + [age]
						old_face.last_centroid = centroid
						### update
						face_objs[saved_object_ids.index(object_id)] = old_face
				#### draw rectangle bounding box for each face
				text = f"ID:{object_id}"
				# print(f'\n===============================')q
				# if gender != 'unknown' and age != 'unknown': print(f"ID:{object_id}--gender {gender}--age {age}")
				# print(f'===============================\n')
				cv2.putText(img=frames,
							text=text,
							org=(centroid[0] - 10, centroid[1] - 10),
							fontFace=cv2.FONT_HERSHEY_SIMPLEX,
							fontScale=0.55,
							color=(0, 255, 0),
							thickness=2)
				cv2.circle(img=frames,
						   center=(centroid[0], centroid[1]),
						   radius=4,
						   color=(0, 255, 0),
						   thickness=1)
			# print(f'len(face_objs): {len(face_objs)}')
			# print(f'current_object_ids: {current_object_ids}')
			for obj in face_objs:
				if obj.id not in current_object_ids:  ### face disappeared
					gender = 'Male' if (obj.gender.count('male') >= obj.gender.count('female')) else 'Female'
					age = max(set(obj.age), key=obj.age.count)
					try:
						going_in = True if obj.first_centroid[-1] > obj.last_centroid[-1] else False

						### remove disappeared object from face_objs and saved face_id
						face_objs.remove(obj)
						saved_object_ids.remove(obj.id)
						# print(f'id: {obj.id}')
						# print(f'gender: {gender}')
						# print(f'age: {age}')
						# print(f'going_in: {going_in}')
						# txt = f'id: {obj.id}\ngender: {gender}\nage: {age}\ngoing_in: {going_in}\n'
						# yield (f'<br><br><br>id: {obj.id}<br>gender: {gender}<br>age: {age}<br>going_in: {going_in}')
						if going_in:
							print(f'Someone is going in')
							send(obj.id, gender, age, going_in)
					except Exception as e:
						face_objs.remove(obj)
						saved_object_ids.remove(obj.id)
						continue
			if args.view_img:
				cv2.imshow('view', frames)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
		except Exception as e:
			print(f'Run_yolo.py 693. Error: {e}')
			cap = VideoCapture(video_path)
			print(f'Recreated video capture object')
			continue
	cv2.destroyAllWindows()


#####################################################################

def detect_in_and_out(yolo_face, yolo_human, age_gender_model, args):
	assert isinstance(age_gender_model, tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
	gender_model, age_model = age_gender_model
	if args.half:
		yolo_face.half()
		yolo_human.half()

	###### for human recognition #########
	# Get names and colors
	names = yolo_human.module.names if hasattr(yolo_human, 'module') else yolo_human.names
	video_path = args.video
	# get stride and image size
	stride = int(yolo_human.stride.max())  # model stride
	img_size = check_img_size(640, s=stride)  # check img_size
	# Set Dataloader
	vid_path, vid_writer = None, None
	#### tracking objects
	face_tracker = CentroidTracker(maxDisappeared=60)
	person_tracker = CentroidTracker(maxDisappeared=60)
	### detected object ids
	saved_face_ids = []
	### person objects list
	faces = []
	### detected object ids
	saved_person_ids = []
	### person objects list
	persons = []
	# dataset = LoadStreams(video_path)
	expand_ratio = 0.0
	dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
	for path, img, im0s, vid_cap in dataset:
		try:
			img = torch.from_numpy(img).to(device).float()
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)
			with torch.no_grad():
				detected_person = yolo_human(img, augment=False)[0]
				detected_face = yolo_face(img, augment=False)[0]
			# Apply NMS
			detected_person = non_max_suppression(detected_person, args.person_score, args.iou, classes=0, agnostic=False)
			detected_face = non_max_suppression(detected_face, args.face_score, args.iou, classes=0, agnostic=False)
			tracking_faces = []
			tracking_persons = []
			# detected_objs = [torch.cat((detected_person[0], detected_face[0]))]


			for i, det in enumerate(detected_face):  # detections per image
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s[0].copy().shape).round()

				# 	# Write resultsqq
				gender, age = 'unknown','unknown'
				for *xyxy, conf, cls in reversed(det):
					if int(cls) == 0:
						#### tracking people in curent frame

						(x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
						tracking_faces.append([x1, y1, x2, y2])
						### expand bounding boxes by expand_ratio
						### x corresponding to column in numpy array --> dim = 1
						x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
						x2 = min(x2 + (expand_ratio * (x2 - x1)), im0s[0].shape[1])
						### y corresponding to row in numpy array --> dim = 0
						y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
						y2 = min(y2 + (expand_ratio * (y2 - y1)), im0s[0].shape[0])
						(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
						### extract the face
						det_img = im0s[0][y1: y2, x1:x2,:].copy()
						# Predict Gender and Age
						with torch.no_grad():
							#### preparing face vector before passing to age, gender estimating model
							vectors = image_loader_(det_img)
							# age_gender_output = age_gender_model(vectors)
							pred_gender = gender_model(vectors)
							pred_age = age_model(vectors)
							## convert predicted index to meaningful label
							gender_indicate = pred_gender.argmax(dim=1).item()
							age_indicate = round(float(pred_age))
							gender = gender_choice.get(gender_indicate)
							age = age_choice.get(age_indicate)
							print(f"detected! {gender} -- {age} years old")
						if args.save_img or args.view_img:  # Add bbox to image
							cv2.rectangle(im0s[0], (x1, y1), (x2, y2), (255, 0, 255), 2)

				### update tracking objects
				face_objects = face_tracker.update(tracking_faces)
				### object ids in current frame (reset each frame)
				current_object_ids = set()
				for (object_id, centroid) in face_objects.items():
					current_object_ids.add(object_id)
					if gender != 'unknown' and age != 'unknown':
						# print(f'there are new object: {object_id}')
						## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
						new_face = Face(id_=object_id, gender=[gender], age=[age], first_centroid=centroid)
						faces.append(new_face)
						saved_face_ids.append(object_id)
					else:
						### when the face object is already in the managing face_objects, update it's info
						### get and edit
						old_person = faces[saved_face_ids.index(object_id)]
						old_person.last_centroid = centroid
						### update
						faces[saved_face_ids.index(object_id)] = old_person
					#### draw rectangle bounding box for each face
					text = f"ID:{object_id}"
					# print(f'\n===============================')
					# print(text)
					# print(f'===============================\n')
					cv2.putText(img=im0s[0],
								text=text,
								org=(centroid[0] - 10, centroid[1] - 10),
								fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								fontScale=0.55,
								color=(0, 255, 0),
								thickness=2)
					cv2.circle(img=im0s[0],
							   center=(centroid[0], centroid[1]),
							   radius=4,
							   color=(0, 255, 0),
							   thickness=1)
				for obj in faces:
					if obj.id not in current_object_ids:  ### face disappeared
						### human recognition model does not have gender and age info
						gender = 'unknown'
						age = -1
						try:
							going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
							### remove disappeared object from face_objs and saved face_id
							faces.remove(obj)
							saved_face_ids.remove(obj.id)
							# print(f'id: {obj.id}')
							# print(f'gender: {gender}')
							# print(f'age: {age}')
							# print(f'going_in: {going_in}')
							# txt = f'id: {obj.id}\ngender: {gender}\nage: {age}\ngoing_in: {going_in}\n'
							# yield (f'<br><br><br>id: {obj.id}<br>gender: {gender}<br>age: {age}<br>going_in: {going_in}')
							if not going_in:
								print(f'Someone is going out')
								send(obj.id, gender, age, going_in)
						except Exception as e:
							faces.remove(obj)
							saved_face_ids.remove(obj.id)
							continue


			for i, det in enumerate(detected_person):  # detections per image
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s[0].copy().shape).round()

				# 	# Write resultsqq
				for *xyxy, conf, cls in reversed(det):
					if int(cls) == 0:
						#### tracking people in curent frame
						(x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
						tracking_persons.append([x1, y1, x2, y2])
						### expand bounding boxes by expand_ratio
						### x corresponding to column in numpy array --> dim = 1
						x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
						x2 = min(x2 + (expand_ratio * (x2 - x1)), im0s[0].shape[1])
						### y corresponding to row in numpy array --> dim = 0
						y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
						y2 = min(y2 + (expand_ratio * (y2 - y1)), im0s[0].shape[0])
						(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))

						if args.save_img or args.view_img:  # Add bbox to image
							cv2.rectangle(im0s[0], (x1, y1), (x2, y2), (255, 0, 0), 2)

				### update tracking objects
				person_objects = person_tracker.update(tracking_persons)
				### object ids in current frame (reset each frame)
				current_object_ids = set()
				for (object_id, centroid) in person_objects.items():
					current_object_ids.add(object_id)

					# print(f'there are new object: {object_id}')
					## when the person  object id is not in saved_person id. put the id into saved_object_id and put person object to person_objs for managing
					new_person = Person(id_=object_id, first_centroid=centroid)
					persons.append(new_person)
					saved_person_ids.append(object_id)

					#### draw rectangle bounding box for each person
					text = f"ID:{object_id}"
					# print(f'\n===============================')
					# print(text)
					# print(f'===============================\n')
					cv2.putText(img=im0s[0],
								text=text,
								org=(centroid[0] - 10, centroid[1] - 10),
								fontFace=cv2.FONT_HERSHEY_SIMPLEX,
								fontScale=0.55,
								color=(0, 255, 0),
								thickness=2)
					cv2.circle(img=im0s[0],
							   center=(centroid[0], centroid[1]),
							   radius=4,
							   color=(0, 255, 0),
							   thickness=1)
				for obj in persons:
					if obj.id not in current_object_ids:  ### person disappeared
						### human recognition model does not have gender and age info
						gender = 'unknown'
						age = -1
						try:
							going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
							### remove disappeared object from (person_objs and saved person_id)
							persons.remove(obj)
							saved_person_ids.remove(obj.id)
							if not going_in:
								print(f'Someone is going out')
								send(obj.id, gender, age, going_in)
						except Exception as e:
							persons.remove(obj)
							saved_person_ids.remove(obj.id)
							continue
				if args.view_img:
					cv2.imshow('view', im0s[0])
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		except NotImplementedError as e:
			print(f'Run_yolo.py 693. Error: {e}')
			continue
	cv2.destroyAllWindows()

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/checkpoints/yolov5s_face.pt', help='path to model weights file')
	parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt', help='path to anchor definitions')
	parser.add_argument('--classes', type=str, default='cfg/face_classes.txt', help='path to class definitions')
	parser.add_argument('--face-score', type=float, default=0.5, help='the face score threshold [0.0 ~ 1.0]')
	parser.add_argument('--person-score', type=float, default=0.45, help='the person score threshold [0.0 ~ 1.0]')
	parser.add_argument('--iou', type=float, default=0.45, help='the iou threshold [0.0 ~ 1.0]')
	parser.add_argument('--img-size', type=list, action='store', default=(416, 416), help='input image size')
	parser.add_argument('--image', default=False, action="store_true", help='image detection mode , boolean type')
	parser.add_argument('--video', type=str, default='rtsp://itaz:12345@192.168.0.33:554/stream_ch00_0', help='path to the video')
	parser.add_argument('--output', default=False, action="store_true", help='whether save the output to video file, boolean type')
	parser.add_argument('--weights', nargs='+', type=str, default='models/checkpoints/yolov5s.pt', help='yolo5 checkpoint path(s)')
	parser.add_argument('--view-img', action='store_true', help='display results, boolean type')
	parser.add_argument('--save-img', action='store_true', help='save results , boolean type')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt , boolean type')
	parser.add_argument('--half', action='store_true', help='running with half precision, boolean type')
	parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	## load arguments
	args = get_args()
	### load models
	# yolov3 = YOLO(args)

	if isinstance(args.device, list):
		device = torch.device(f'cuda:{int(id_)}' for id_ in args.device)
	elif isinstance(args.device, int):
		device = torch.device(f'cuda:{int(args.device)}')
	else:
		device = torch.device('cpu')
	yolo_face = attempt_load(args.model, map_location=device)
	yolo_human = attempt_load(args.weights, map_location=device)
	age_gender_model = get_model()
	# detect_img(yolov3, r'G:\locs_projects\on_working\images\test_images', age_gender_model)
	# detect_img(yolov3, r'val_images', age_gender_model)
	detect_in_and_out(yolo_face, yolo_human, age_gender_model, args)
## run 2 models on separate threads
# run_in = threading.Thread(target=detect_in, args=(yolov3, age_gender_model, args))
# # run_out = threading.Thread(target=detect_out, args=(yolov5, args))
#
# # run_out.start()
# run_in.start()
# # detect_out(yolov5, args)
# # run_out.join()
# run_in.join()
