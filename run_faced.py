from cv2 import cv2

from faced import FaceDetector
from faced.utils import annotate_image

#
#
# img = cv2.imread(r'C:\Users\locs\Desktop\New folder\Detection_Face_Information\image\133-0.jpg')
# rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
#
# # Receives RGB numpy image (HxWxC) and
# # returns (x_center, y_center, width, height, prob) tuples.
# bboxes = face_detector.predict(rgb_img)
#
# # Use this utils function to annotate the image.
# ann_img = annotate_image(img, bboxes)
#
# # Show the image
# cv2.imshow('image',ann_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#

from cv2 import cv2
from hparams import *
import numpy as np
import os
import matplotlib.pyplot as plt
# cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml" ## get classifier from cv2 data folder
import face_recognition

### get pretrained model of age and gender detection
def load_caffe_models():
	age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
	gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
	return (age_net, gender_net)


def video_detector(age_net, gender_net):
	### get face detection params
	face_detector = FaceDetector()

	### define video stream source
	cap = cv2.VideoCapture('http://164.125.154.221:8090/?action=stream')
	## resize video frame
	cap.set(3, 480)  # set width of the frame
	cap.set(4, 640)  # set height of the frame

	## save output

	video_fourcc = cv2.VideoWriter_fourcc('M', 'G', 'P', 'G')
	video_fps = cap.get(cv2.CAP_PROP_FPS)
	# the size of the frames to write
	video_size = (int(640),int(480))

	output_fn = 'output_video.mp4'
	out = cv2.VideoWriter(os.path.join('outputs/', output_fn), video_fourcc, video_fps, video_size)



	while True:
		# Capture frame-by-frame
		ret, frames = cap.read()
		gray = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
		### adjust constrast and brightness
		gray = cv2.convertScaleAbs(gray, alpha=0.98, beta=1)

		### detect faces using haarcascade
		# faces = faceCascade.detectMultiScale(gray,scaleFactor=1.09,minNeighbors=5, minSize=(8,8))

		## detect faces using face-recognition library.
		faces = face_detector.predict(gray, thresh=0.88)
		if (len(faces) > 0):
			print("Found {} faces".format(str(len(faces))))

		# Draw a rectangle around the faces
		overlay_texts=[]
		for (x, y, w, h, score) in faces:
			# cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 255, 0), 2)

			### extract the face
			face_img = frames[y:y + h, h:h + w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
			# Predict Gender
			gender_net.setInput(blob)
			gender_preds = gender_net.forward()
			print(gender_preds)
			gender = gender_list[gender_preds[0].argmax()]
			print("Gender : " + gender)  # Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax()]
			print("Age Range: " + age)
			overlay_text = "%s, %s, %s" % (gender, age, score)
			overlay_texts.append(overlay_text)
			# cv2.putText(frames, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

		frames = annotate_image(frames, faces, overlay_texts)

		# Display the resulting frame
		cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
		cv2.imshow('Video', frames)
		out.write(frames)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

if __name__=='__main__':
	age_net, gender_net = load_caffe_models()
	video_detector(age_net, gender_net)

