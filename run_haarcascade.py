import datetime
import os
import time
import cv2
# cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml" ## get classifier from cv2 data folder
import torch

from hparams import *

sex_choice={1:'Male', 2:'Female'}
age_choice={1:'<10', 2:'10~19', 3:'20~29', 4:'30~39', 5:'40~49', 6:'50~59', 7:'60~69', 8:'70~79', 9:'80~89', 10:'>=90'}


### get pretrained model of age and gender detection
def get_model():
	model_path = 'models/vgg19-200-0.01_051200_1592307399.28784.pth'
	checkpoint = torch.load(model_path, map_location='cuda:0')
	model_type = checkpoint['model_type']
	model_parameter = checkpoint['model_parameter']
	m = None
	if model_type == 'vgg':
		from modules.vgg import VGG
		m = VGG(**model_parameter)

	elif model_type == 'fr_net':
		from modules.face_recognition import FRNet
		m = FRNet()
	m.load_state_dict(checkpoint['model_state_dict'])
	m.eval()
	return m


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

age_gender_model = get_model()

def video_detector():
	### get face detection params
	cascPath = "haarcascade_frontalface_default.xml"  ## get classifier from this folder
	faceCascade = cv2.CascadeClassifier(cascPath)

	### define video stream source
	cap = cv2.VideoCapture('http://164.125.154.221:8090/?action=stream')
	## resize video frame
	cap.set(3, 480)  # set width of the frame
	cap.set(4, 640)  # set height of the frame

	## save output

	# video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	# video_fps = cap.get(cv2.CAP_PROP_FPS)
	# the size of the frames to write
	video_size = (int(640),int(480))

	output_fn = 'output_video.mp4'
	video_fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
	out = cv2.VideoWriter((os.path.join('outputs/', output_fn), video_fourcc, 20.0, video_size))
	while True:
		# Capture frame-by-frame
		ret, frames = cap.read()
		out.write(frames)
		### adjust constrast and brightness
		# gray = cv2.convertScaleAbs(gray, alpha=0.5, beta=100)
		### detect faces using haarcascade
		# faces = faceCascade.detectMultiScale(gray,scaleFactor=1.09,minNeighbors=5, minSize=(8,8))

		faces = faceCascade.detectMultiScale(frames, scaleFactor=1.09, minNeighbors=5, minSize=(8, 8))
		if (len(faces) > 0):
			print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h) in faces:
			### detected face will be extend using this (we want see the face widerly, not only face, might be shoulder, hair and neck)
			extend = 0#int(w * 0.2)

			### if the image is not large enough to extend (when the faces are so close to camera, or are in the coner) set extend to 0
			if extend > y or extend >x or extend > frames.shape[0]-(x+w) or extend > frames.shape[1] -(y+h):
				extend = 0
			print(extend)
			### extract the face
			face_img = frames[y - extend:y+h+ extend, x- extend:x+w+ extend].copy()


			#### preparing face vector before passing to age, gender estimating model
			vectors = image_loader_(face_img)
			# Predict Gender and Age
			age_gender_output = age_gender_model(vectors)

			sex_indicate = [i + 1 for i in age_gender_output['sex'].argmax(dim=-1).tolist()]
			age_indicate = [i + 1 for i in age_gender_output['age'].argmax(dim=-1).tolist()]
			overlay_text = "%s, %s" % (sex_choice.get(sex_indicate[0]), age_choice.get(age_indicate[0]))
			# Draw a rectangle around the faces
			cv2.rectangle(frames, (x - extend, y - extend), (x + w + extend, y + h + extend), (0, 255, 0), 2)
			cv2.putText(frames, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			print(f"시점: {str(datetime.datetime.now().time())} -- 성별: {sex_choice.get(sex_indicate[0])} -- 나이: {age_choice.get(age_indicate[0])}\n")

			### save face image for debugging
			img_path = f'templates\\{overlay_text}_{time.time()}.jpg'
			cv2.imwrite(img_path, face_img)
		# Display the resulting frame
		try:
			cv2.imshow('Video', frames)

		except:
			continue
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	# out.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	video_detector()

