import argparse
from random import randint
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import time
from cv2 import cv2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from modules.new_yolo import YOLO
import uuid





def get_faces_path(path):
	paths = []
	print(f'Loading image paths.. \n')
	for dir_path,last_dir,files in tqdm(os.walk(path)):
		for file in files:
			if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
				paths.append(os.path.join(dir_path,file))
	return paths


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
	parser.add_argument('--model', type=str, default='models/YOLO_Face.h5',
						help='path to model weights file')
	parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt',
						help='path to anchor definitions')
	parser.add_argument('--classes', type=str, default='cfg/face_classes.txt',
						help='path to class definitions')
	parser.add_argument('--score', type=float, default=0.5,
						help='the score threshold')
	parser.add_argument('--iou', type=float, default=0.45,help='the iou threshold')
	parser.add_argument('--img-size', type=list, action='store',
						default=(416, 416), help='input image size')
	parser.add_argument('--image', default=False, action="store_true",
						help='image detection mode')
	parser.add_argument('--video', type=str, default='http://119.198.38.200:8090/?action=stream',
						help='path to the video')
	parser.add_argument('--output', type=str, default='1',
						help='image/video output path')
	args = parser.parse_args()
	return args


if __name__=='__main__':

	### load ai hub data and put them into dataloader
	path = r'dataset/High_resolution'
	# path = r'G:\data\images\faces\AIhub_face\High_resolution'
	paths_ = get_faces_path(path)



	### use this to extract only dark faces (faces in L25 ~ L30 folder)####
	to_get = [f'L{i}' for i in range(25, 31)]
	paths = []
	### exclude path does not contain L25~L30 folder
	for pth in paths_:
		for folder_ in to_get:
			if folder_ in pth:
				print(pth)
				paths.append(pth)
	#######################################################################




	dataset = ImageDataset(paths)
	print(len(dataset))
	batch_size=32
	image_dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size, num_workers=1)

	args = get_args()
	yolo_model = YOLO(args=args)
	#### load yolo model
	batch_size = None
	out_dir = 'dataset/extracted_darkfaces'
	os.makedirs(out_dir,exist_ok=True)

	for (image, age, gender, image_path) in image_dataloader:
		batch_size = image.size(0)
		for i in range(batch_size):
			original_img = image[i].data.numpy()
			img = Image.fromarray(original_img)
			###### check image before passing to model
			img, faces = yolo_model.detect_image(img)

			for (x1, y1, x2, y2) in faces:  ## with yolo, result will be 2 point of rectangle corner (x1, y1) and (x2, y2)
				try:
					### extend the region to get wider face
					### limit the x1,y1 to make sure they are >0.
					### similar to x2,y2, make sure they are <= image max height and width
					x1 = max(x1 - (0.2*(x2-x1)),0)
					x2 = min(x2 + (0.2*(x2-x1)),original_img.shape[0])
					y1 = max(y1 - (0.2*(y2-y1)),0)
					y2 = min(y2 + (0.2*(y2-y1)),original_img.shape[1])
					## get size of face image
					(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
					face_img = original_img[x1: x2, y1:y2].copy()
					### extract the face (228, 205, 3)
					_id = uuid.uuid1().int
					# print(f'{image_path[i]} -- {int(age[i])} -- {gender[i]} -- {_id}')
					cv2.imwrite(os.path.join(out_dir,f'{_id}_{int(age[i])}_{gender[i]}.jpg'), face_img)
					print(os.path.join(out_dir,f'{_id}_{int(age[i])}_{gender[i]}.jpg'))
				except Exception:
					raise Exception('Error')

