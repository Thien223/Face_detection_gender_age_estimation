import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFile
from  random import randint
import face_recognition

DEFAULT_IMAGE_PATH = 'dataset/mask_images/default-mask.png'
BLACK_IMAGE_PATH = 'dataset/mask_images/black-mask.png'
BLUE_IMAGE_PATH = 'dataset/mask_images/blue-mask.png'
RED_IMAGE_PATH = 'dataset/mask_images/red-mask.png'
KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
# mask_paths = [DEFAULT_IMAGE_PATH,BLUE_IMAGE_PATH,BLACK_IMAGE_PATH,RED_IMAGE_PATH]
mask_paths = [DEFAULT_IMAGE_PATH,BLUE_IMAGE_PATH]

def mask(face_path, mask_path=DEFAULT_IMAGE_PATH,show=False, model='hog'):
	face_image_np = face_recognition.load_image_file(face_path)
	face_locations = face_recognition.face_locations(face_image_np, model=model)
	face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
	_face_img = Image.fromarray(face_image_np)
	_mask_img = Image.open(mask_path)


	found_face = False
	for face_landmark in face_landmarks:
		## check whether facial features meet requirement
		skip = False
		for facial_feature in KEY_FACIAL_FEATURES:
			if facial_feature not in face_landmark:
				skip = True
				break
		if skip:
			continue

		# mask face
		found_face = True
		_mask_face(face_landmark, _mask_img, _face_img)

	if found_face:
		if show:
			_face_img.show()
		# save
		_save(face_path, _face_img)
	else:
		_mask_img.close()
		print('Found no face.')
	return 0

def _mask_face(face_landmark: dict, _mask_img, _face_img):
	nose_bridge = face_landmark['nose_bridge']
	nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
	nose_v = np.array(nose_point)
	#####
	chin = face_landmark['chin']
	chin_len = len(chin)
	chin_bottom_point = chin[chin_len // 2]
	chin_bottom_v = np.array(chin_bottom_point)
	chin_left_point = chin[chin_len // 8]
	chin_right_point = chin[chin_len * 7 // 8]

	## split mask and resize
	width = _mask_img.width
	height = _mask_img.height
	width_ratio = 1.2
	new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

	## left
	mask_left_img = _mask_img.crop((0, 0, width // 2, height))
	mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
	mask_left_width = int(mask_left_width * width_ratio)
	mask_left_img = mask_left_img.resize((mask_left_width, new_height))

	## right
	mask_right_img = _mask_img.crop((width // 2, 0, width, height))
	mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
	mask_right_width = int(mask_right_width * width_ratio)
	mask_right_img = mask_right_img.resize((mask_right_width, new_height))

	# merge mask
	size = (mask_left_img.width + mask_right_img.width, new_height)
	mask_img = Image.new('RGBA', size)
	mask_img.paste(mask_left_img, (0, 0), mask_left_img)
	mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

	# rotate mask
	angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
	rotated_mask_img = mask_img.rotate(angle, expand=True)

	# calculate mask location
	center_x = (nose_point[0] + chin_bottom_point[0]) // 2
	center_y = (nose_point[1] + chin_bottom_point[1]) // 2

	offset = mask_img.width // 2 - mask_left_img.width
	radian = angle * np.pi / 180
	box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
	box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2
	# add mask
	_face_img.paste(mask_img, (box_x, box_y), mask_img)


def _save(face_path, _face_img):
	path_splits = os.path.splitext(face_path)
	new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
	_face_img.save(new_face_path)
	print(f'Saved: {new_face_path}')

def get_distance_from_point_to_line(point, line_point1, line_point2):
	distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
					  (line_point1[0] - line_point2[0]) * point[1] +
					  (line_point2[0] - line_point1[0]) * line_point1[1] +
					  (line_point1[1] - line_point2[1]) * line_point1[0]) / \
			   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
					   (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
	return int(distance)




def main():
	parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
	parser.add_argument('--pic_path', default='d:\\', help='Picture path.')
	parser.add_argument('--recursively', default=False, action='store_true', help='whether exploring folder recursively')
	args = parser.parse_args()

	pic_path = args.pic_path
	# if not os.path.exists(args.pic_path):
	# 	print(f'Picture {pic_path} not exists.')
	# 	return
	#
	if args.recursively:
		for (dir,_,files) in os.walk(pic_path):
			for f in files:
				img_path = os.path.join(dir,f)
				if img_path.endswith('.jpg') or img_path.endswith('.png') or img_path.endswith('.jpeg'):
					mask_path = mask_paths[randint(0, len(mask_paths)-1)]
					try:
						mask(img_path, mask_path)
					except ValueError as e:
						print(e)
						continue
				else:
					pass
	else:
		for file in os.listdir(pic_path):
			mask_path = mask_paths[randint(0, len(mask_paths) - 1)]
			img_path = os.path.join(pic_path, file)
			if img_path.endswith('.jpg') or img_path.endswith('.png') or img_path.endswith('.jpeg'):
				try:
					mask(img_path, mask_path)
				except ValueError as e:
					print(e)
					continue


if __name__=='__main__':
	main()