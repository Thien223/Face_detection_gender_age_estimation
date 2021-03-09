### thien.locslab@gmail.com
import argparse

import cv2
import numpy as np
import torch
# from flask import Flask, render_template, request, stream_with_context

from modules.new_yolo import YOLO, detect_video
#
# UPLOAD_FOLDER = 'static/data/'
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

### get pretrained model of age and gender detection
def get_model():
	# model_path = 'models/vgg19-epochs_100-step_68400-gender_accuracy_0.9896511380062547.pth'
	model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
	checkpoint = torch.load(model_path, map_location='cpu')
	model_type = checkpoint['model_type']
	model_parameter = checkpoint['model_parameter']
	m = None
	if model_type == 'vgg':
		from modules.vgg import VGG
		m = VGG(**model_parameter)

	elif model_type == 'fr_net':
		from modules.frnet import FRNet
		m = FRNet()
	m.load_state_dict(checkpoint['model_state_dict'])
	m.eval()
	return m


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
	parser.add_argument('--video', type=str, default='http://175.200.252.169:8090/?action=stream',
						help='path to the video')
	parser.add_argument('--output', type=str, default='1',
						help='image/video output path')
	args = parser.parse_args()
	return args

def transform_to_bird_eye_view(source, original_object_points, target_object_points):

	src = np.float32([[original_object_points[0][0], original_object_points[8][0], original_object_points[45][0], original_object_points[53][0]]])
	dst = np.float32([target_object_points[0][0], target_object_points[8][0], target_object_points[45][0], target_object_points[53][0]])
	transformer = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix
	# inverse_transformer = cv2.getPerspectiveTransform(dst, src)  # Inverse transformation

	vid = cv2.VideoCapture(source)
	offset=200
	while True:
		ret, img = vid.read()
		warped_img = cv2.warpPerspective(img, transformer, (img.shape[0] + offset, img.shape[1] + offset))  # Image warping

		cv2.imshow("output", warped_img*0)  # Show results
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



#
# @app.route('/')
# def index():
# 	return render_template('index.html')
#
#
# @app.route('/stream', methods = ['GET', 'POST'])
# def detect_camera():
# 	age_gender_model = get_model()
# 	args = get_args()
# 	detect_video(YOLO(args), args.video , age_gender_model)
#
# @app.route('/video', methods = ['GET', 'POST'])
# def video():
# 	headers = dict()
# 	headers['Access-Control-Allow-Origin'] = '*'
# 	age_gender_model = get_model()
# 	args = get_args()
# 	import os
# 	file = request.files['file']
# 	if (file.filename.endswith('.mp4') or file.filename.endswith('.avi')):
# 		### run code here
# 		file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
# 		file.save(file_path)
#
#
# 		return app.response_class(stream_with_context(detect_video(YOLO(args), file_path , age_gender_model)),headers=headers, mimetype='text/event-stream')
# 	else:
#
# 		return app.response_class('', mimetype='text/event-stream')


if __name__ == "__main__":

	#
	# print(("Starting server..."))
	# app.run(host='0.0.0.0', port=80,threaded=True)
	age_gender_model = get_model()
	args = get_args()
	detect_video(YOLO(args),args.video,age_gender_model)
