import argparse
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import time
from cv2 import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from modules.new_yolo import YOLO, attempt_load
import uuid

from utils.datasets import letterbox
from utils.general import check_img_size, scale_coords, non_max_suppression


torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
np.random.seed(1542)
def get_faces_path(path):
	paths = []
	print(f'Loading image paths.. \n')
	for dir_path,last_dir,files in tqdm(os.walk(path)):
		for file in files:
			if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
				paths.append(os.path.join(dir_path,file))
	return paths

# image_path = r'F:\\data\\images\\faces\\AIhub_face\\High_resolution\\19070231\\S006\\L30\\C9.jpg'
def get_age(image_path):
	# assert isinstance(str,image_path),'image path must be string..'
	import pandas as pd
	age_mapping_df =pd.read_excel(r'dataset/KFace_data_information_Folder1_400.xlsx', engine='openpyxl')
	for row in age_mapping_df.itertuples():
		id = str(row[1])
		if id in image_path:
			age = row[2].replace('대','').strip()
			gender = 'male' if row[3]=='남' else 'female'
			return int(age), gender
		else:
			continue


class ImageDataset(Dataset):
	def __init__(self, img_folder):
		super(ImageDataset, self).__init__()
		self.image_path = img_folder

	def __getitem__(self, index_):
		image_path = self.image_path[index_]
		age, gender = get_age(image_path)
		image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

		return image_np, age, gender, image_path

	def __len__(self):
		return len(self.image_path)



#####################################################################
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='models/checkpoints/yolov5s_face.pt',
						help='path to model weights file')
	parser.add_argument('--score', type=float, default=0.5,
						help='the score threshold')
	parser.add_argument('--iou', type=float, default=0.45,help='the iou threshold')
	parser.add_argument('--img-size', type=list, action='store',
						default=(416, 416), help='input image size')
	parser.add_argument('--image', default=False, action="store_true",
						help='image detection mode')
	parser.add_argument('--device', type=int, default=0,
						help='GPU device to run')
	parser.add_argument('--brightness-limit', type=int, default=10,
						help='GPU device to run')
	args = parser.parse_args()
	return args


if __name__=='__main__':

	### load ai hub data and put them into dataloader
	path = r'dataset/aihub'
	# path = r'G:\data\images\faces\AIhub_face\High_resolution'
	paths_ = get_faces_path(path)
	args = get_args()



	### use this to extract only faces with brightness condition
	to_get = [f'L{i}' for i in range(args.brightness_limit)]
	paths = []
	### exclude path does not match the condition above
	for pth in tqdm(paths_):
		for folder_ in to_get:
			if folder_ in pth:
				paths.append(pth)
	#######################################################################




	dataset = ImageDataset(paths)
	batch_size = 64
	image_dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size, num_workers=1)

	device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
	# yolo_model = YOLO(args=args)
	yolo_face = attempt_load(args.model, map_location=device)
	stride = int(yolo_face.stride.max())  # model stride
	img_size = check_img_size(640, s=stride)  # check img_size


	out_dir = 'dataset/extracted_aihub'
	os.makedirs(out_dir,exist_ok=True)
	for images, ages, genders, image_paths in tqdm(image_dataloader):
		for ii, (_, age, gender, img_p) in enumerate(zip(images,ages,genders,image_paths)):
			# print(ii)
			frames = cv2.imread(img_p)
			_id = uuid.uuid1().int
			p = f'{_id}_{img_p.split("/")[-5]}_{int(age)}_{str(gender)}.jpg'

			# print('')
			# print(age[b])
			# print(image_path[b].split("/")[-5])
			# print(gender[b])
			# print('')
			#
			img = [letterbox(x, img_size, auto=True, stride=stride)[0] for x in [frames]]
			# STACK
			img = np.stack(img, axis=0)

			# Convert img
			img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # ToTensor image convert to BGR, RGB
			img = np.ascontiguousarray(img)  # frame imgs, 메모리에 연속 배열 (ndim> = 1)을 반환
			img = torch.from_numpy(img).to(device)  # numpy array convert
			# print(img)
			img = img.float()  # unit8 to 16/32
			img /= 255.0  # 0~255 to 0.0~1.0 images.

			if img.ndimension() == 3:
				img = img.unsqueeze(0)

			detected_face = yolo_face(img, augment=False)[0]
			detected_face = non_max_suppression(detected_face, 0.6, 0.45, classes=0, agnostic=False)
			# if len(detected_face)<1:
			# 	print("cannot detected any face")
			# 	continue

			expand_ratio=0.05
			det = detected_face[0]  # detections per image
			# Rescale boxes from img_size to frames size
			det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()
			if len(reversed(det)) < 1:
				continue
			else:

				for *xyxy, conf, cls in reversed(det):
					# print(f'{ii} === xyxy {xyxy}')
					#### tracking people in curent frame
					(x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
					### expand bounding boxes by expand_ratio
					### x corresponding to column in numpy array --> dim = 1
					x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
					x2 = min(x2 + (expand_ratio * (x2 - x1)), frames.shape[1])
					### y corresponding to row in numpy array --> dim = 0
					y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
					y2 = min(y2 + (expand_ratio * (y2 - y1)), frames.shape[0])
					(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
					### extract the face
					det_img = frames[y1: y2, x1:x2, :].copy()
					cv2.imwrite(os.path.join(out_dir, p), det_img)
					break