### thien.locslab@gmail.com
import argparse
import os
import random

import numpy as np
# import tensorflow as tf
import torch
from cv2 import cv2

from modules.new_yolo import Face, Person, attempt_load, VideoCapture
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
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
    data = dict(query=str_, variables={})
    print(data)
    requests.post(URL, data=json.dumps(data), headers=headers)


### get pretrained model of age and gender detection
def get_model(model_path=None):
    if model_path is None:
        model_path = 'models/checkpoints/vgg-epochs_464-step_0-gender_acc_98.5440541048281-age_acc_83.13920721397709.pth'
    gender_model_path = 'models/checkpoints/vgg19-epochs_638-step_0-gender_accuracy_0.9546627402305603.pth'
    # gender_model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_2 = torch.load(gender_model_path, map_location=device)
    model_type = checkpoint['model_type']

    weights = {}
    for k, v in checkpoint_2['model_state_dict'].items():
        k = k.replace('sex', 'gender')
        weights[k] = v

    if model_type == 'vgg':
        from modules.vgg import VGG, Age_VGG
        gender_model = VGG(vgg_type='vgg19')
        age_model = Age_VGG(vgg_type='vgg19')
        gender_model.load_state_dict(weights)
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


#####################################################################

def detect_in_and_out(yolo_face, yolo_human, age_gender_model, args):
    assert isinstance(age_gender_model,
                      tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
    gender_model, age_model = age_gender_model

    ###### for human recognition #########
    # Get names and colors
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
    is_webcam = video_path.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    # if is_webcam:
    # 	import torch.backends.cudnn as cudnn
    # 	cudnn.benchmark = True
    # 	dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
    # else:
    # 	dataset = LoadImages(video_path, img_size=img_size, stride=stride)

    cap = VideoCapture(video_path)

    while True:
        frames = cap.read()

        if frames is None or cap is None:
            cap = VideoCapture(video_path)
            # cap = cv2.VideoCapture(video_path)
            print(f'------------------ recreated video capture')
            continue
        else:
            # print(f'processing video! - out')
            # letterbox() : image Resize and pad image while meeting stride-multiple constraints
            # auto -> padding minimum rectangle
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

            with torch.no_grad():
                detected_person = yolo_human(img, augment=False)[0]
                detected_face = yolo_face(img, augment=False)[0]
                # Apply NMS
                detected_person = non_max_suppression(detected_person, args.person_score, args.iou, classes=0,
                                                      agnostic=False)
                detected_face = non_max_suppression(detected_face, args.face_score, args.iou, classes=0, agnostic=False)
                tracking_faces = []
                tracking_persons = []
            # detected_objs = [torch.cat((detected_person[0], detected_face[0]))]

            for i, det in enumerate(detected_face):  # detections per image
                # Rescale boxes from img_size to frames size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()

                # 	# Write resultsqq
                gender, age = 'unknown', 'unknown'
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:
                        #### tracking people in curent frame
                        # (x1, y1, x2, y2) = (int(xyxy[0]) * 2, int(xyxy[1]) * 2, int(xyxy[2]) * 2, int(xyxy[3]) * 2)
                        (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        tracking_faces.append([x1, y1, x2, y2])
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

                        # Predict Gender and Age
                        with torch.no_grad():
                            #### preparing face vector before passing to age, gender estimating model
                            vectors = image_loader_(det_img)
                            # age_gender_output = age_gender_model(vectors)
                            pred_gender = gender_model(vectors)
                            pred_age = age_model(vectors)
                            ## convert predicted index to meaningful label
                            gender_indicate = pred_gender['gender'].argmax(dim=1).item()
                            age_indicate = round(float(pred_age))
                            gender = gender_choice.get(gender_indicate)
                            age = age_choice.get(age_indicate)
                        # print(f"detected! {gender} -- {age} years decade")
                        if args.save_img or args.view_img:  # Add bbox to image
                            cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 255), 2)

                ### update tracking objects
                face_objects = face_tracker.update(tracking_faces)
                ### object ids in current frame (reset each frame)
                current_object_ids = set()
                for (object_id, centroid) in face_objects.items():
                    current_object_ids.add(object_id)
                    if object_id not in saved_face_ids:
                        if gender != 'unknown' and age != 'unknown':
                            # print(f'there are new object: {object_id}')
                            ## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
                            new_face = Face(id_=object_id, gender=[gender], age=[age], first_centroid=centroid)
                            faces.append(new_face)
                            saved_face_ids.append(object_id)
                        # print(f'len(faces) {len(faces)}')
                    else:
                        if gender != 'unknown' and age != 'unknown':
                            ### when the face object is already in the managing face_objects, update it's info
                            ### get and edit
                            old_face = faces[saved_face_ids.index(object_id)]
                            old_face.gender = old_face.gender + [gender]
                            old_face.age = old_face.age + [age]
                            old_face.last_centroid = centroid
                            ### update
                            faces[saved_face_ids.index(object_id)] = old_face
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
                for obj in faces:
                    if obj.id not in current_object_ids:  ### face disappeared
                        gender = 'Male' if (obj.gender.count('male') >= obj.gender.count('female')) else 'Female'
                        age = max(set(obj.age), key=obj.age.count)
                        try:
                            going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
                            ### remove disappeared object from face_objs and saved face_id
                            if going_in:
                                print(f'Someone is going in')
                                try:
                                    send(obj.id, gender, age, going_in)
                                except ConnectionError as e:
                                    print(
                                        f'sending data to database was failed. Check the database server connection..')
                                    continue
                            faces.remove(obj)
                            saved_face_ids.remove(obj.id)

                        except Exception as e:
                            if obj in faces:
                                faces.remove(obj)
                            if obj.id in saved_face_ids:
                                saved_face_ids.remove(obj.id)
                            continue

            for i, det in enumerate(detected_person):  # detections per image
                # Rescale boxes from img_size to frames size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()

                # 	# Write resultsqq
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:
                        #### tracking people in curent frame
                        (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        tracking_persons.append([x1, y1, x2, y2])
                        ### expand bounding boxes by expand_ratio
                        ### x corresponding to column in numpy array --> dim = 1
                        x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
                        x2 = min(x2 + (expand_ratio * (x2 - x1)), frames.shape[1])
                        ### y corresponding to row in numpy array --> dim = 0
                        y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
                        y2 = min(y2 + (expand_ratio * (y2 - y1)), frames.shape[0])
                        (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                        if args.save_img or args.view_img:  # Add bbox to image
                            cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 0), 2)

                ### update tracking objects
                person_objects = person_tracker.update(tracking_persons)
                ### object ids in current frame (reset each frame)
                current_object_ids = set()
                for (object_id, centroid) in person_objects.items():
                    current_object_ids.add(object_id)
                    if object_id not in saved_person_ids:
                        # print(f'there are new object: {object_id}')
                        ## when the person  object id is not in saved_person id. put the id into saved_object_id and put person object to person_objs for managing
                        new_person = Person(id_=object_id, first_centroid=centroid)
                        persons.append(new_person)
                        saved_person_ids.append(object_id)
                    else:
                        ### when the face object is already in the managing face_objects, update it's info
                        ### get and edit
                        old_person = persons[saved_person_ids.index(object_id)]
                        old_person.last_centroid = centroid
                        ### update
                        persons[saved_person_ids.index(object_id)] = old_person

                    #### draw rectangle bounding box for each person
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
                for obj in persons:
                    if obj.id not in current_object_ids:  ### person disappeared
                        ### human recognition model does not have gender and age info
                        gender = 'unknown'
                        age = -1
                        try:
                            going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
                            ### remove disappeared object from (person_objs and saved person_id)
                            if not going_in:
                                print(f'Someone is going out')
                                try:
                                    send(obj.id, gender, age, going_in)
                                except ConnectionError as e:
                                    print(
                                        f'sending data to database was failed. Check the database server connection..')
                                    continue
                            persons.remove(obj)
                            saved_person_ids.remove(obj.id)

                        except Exception as e:
                            if obj in persons:
                                persons.remove(obj)
                            if obj.id in saved_person_ids:
                                saved_person_ids.remove(obj.id)
                            continue
                if args.view_img:
                    cv2.imshow('view', frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/checkpoints/yolov5s_face.pt',
                        help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt', help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='cfg/face_classes.txt', help='path to class definitions')
    parser.add_argument('--face-score', type=float, default=0.5, help='the face score threshold [0.0 ~ 1.0]')
    parser.add_argument('--person-score', type=float, default=0.45, help='the person score threshold [0.0 ~ 1.0]')
    parser.add_argument('--iou', type=float, default=0.45, help='the iou threshold [0.0 ~ 1.0]')
    parser.add_argument('--img-size', type=list, action='store', default=(416, 416), help='input image size')
    parser.add_argument('--image', default=False, action="store_true", help='image detection mode , boolean type')
    parser.add_argument('--video', type=str, default='rtsp://itaz:12345@192.168.0.33:554/stream_ch00_0',
                        help='path to the video')
    parser.add_argument('--output', default=False, action="store_true",
                        help='whether save the output to video file, boolean type')
    parser.add_argument('--weights', nargs='+', type=str, default='models/checkpoints/yolov5s.pt',
                        help='yolo5 checkpoint path(s)')
    parser.add_argument('--view-img', action='store_true', help='display results, boolean type')
    parser.add_argument('--save-img', action='store_true', help='save results , boolean type')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt , boolean type')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    ## load arguments
    args = get_args()
    ### load models
    # yolov3 = YOLO(args)

    if isinstance(args.device, list):
        device = torch.device(f'cuda:{int(args.device)}')
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


# ### thien.locslab@gmail.com
# import argparse
# import os
# import random
# import time
# from pathlib import Path
# import threading
# import numpy as np
# # import tensorflow as tf
# import torch
# from PIL import Image
# from cv2 import cv2
#
# from modules.new_yolo import Face, Person, YOLO, attempt_load, VideoCapture
# from train_module.data_helper import _age_categorization
# from utils.datasets import LoadStreams, LoadImages, letterbox
# from utils.general import check_img_size, non_max_suppression, scale_coords, \
# 	xyxy2xywh, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import time_synchronized
# from utils.trac_object import CentroidTracker
#
# torch.manual_seed(1542)
# torch.cuda.manual_seed(1542)
# torch.backends.deterministic = True
# torch.backends.benchmark = False
# random.seed(1542)
# np.random.seed(1542)
# # tf.random.set_seed(1542)
# # tf.compat.v1.disable_eager_execution()
#
# os.environ['KERAS_BACKEND'] = 'tensorflow'
#
# gender_choice = {0: 'female', 1: 'male'}
# age_choice = {0: '10', 1: '20', 2: '30', 3: '40', 4: '50'}
#
#
#
# def image_loader_(face_img):
# 	from PIL import Image
# 	from torchvision import transforms
# 	import numpy as np
# 	from train_module.data_helper import img_size  #### img_size=64
# 	# image_set = []
# 	preprocess = transforms.Compose([transforms.Resize(img_size + 4),
# 									 transforms.RandomCrop(img_size),
# 									 transforms.RandomHorizontalFlip(),
# 									 transforms.ToTensor(),
# 									 # transforms.Normalize(mean=mean,
# 									 #                      std=std)
# 									 ])
# 	# face_img = Image.fromarray(np.uint8(face_img))
# 	face_img = Image.fromarray(np.uint8(face_img))
# 	# face_img = Image.open(face_img)
# 	img = preprocess(face_img)
# 	img = img.unsqueeze(dim=0)
# 	return img
#
#
# def image_loader(face_img_path):
# 	from PIL import Image
# 	from train_module.data_helper import image_preprocess
# 	# image_set = []
# 	preprocess = image_preprocess(img_path=None)
# 	face_img = Image.open(face_img_path)
# 	img = preprocess(face_img)
# 	img = img.unsqueeze(dim=0)
# 	return img
#
#
# def send(id, gender, age, going_in):
# 	import requests
# 	import json
# 	URL = 'http://museum.locslab.com/api'
# 	headers = {
# 		'Content-Type': 'application/json',
# 	}
# 	str_ = 'mutation{createDetection(identifier:%d, age:\"%s\", gender:\"%s\", inAndOut:%s){identifier age gender inAndOut}}' % (
# 		id, age, gender, str(going_in).lower())
# 	data = dict(query=str_, variables={})
# 	print(data)
# 	requests.post(URL, data=json.dumps(data), headers=headers)
#
#
# ### get pretrained model of age and gender detection
# def get_model(model_path=None):
# 	if model_path is None:
# 		model_path = 'models/checkpoints/vgg-epochs_464-step_0-gender_acc_98.5440541048281-age_acc_83.13920721397709.pth'
# 	gender_model_path = 'models/checkpoints/vgg19-epochs_638-step_0-gender_accuracy_0.9546627402305603.pth'
# 		# gender_model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
# 	checkpoint = torch.load(model_path, map_location=device)
# 	checkpoint_2 = torch.load(gender_model_path, map_location=device)
# 	model_type = checkpoint['model_type']
#
# 	weights = {}
# 	for k,v in checkpoint_2['model_state_dict'].items():
# 		k=k.replace('sex','gender')
# 		weights[k] = v
#
# 	if model_type == 'vgg':
# 		from modules.vgg import VGG, Age_VGG
# 		gender_model = VGG(vgg_type='vgg19')
# 		age_model = Age_VGG(vgg_type='vgg19')
# 		gender_model.load_state_dict(weights)
# 		age_model.load_state_dict(checkpoint['age_model_weights'])
# 		gender_model.eval()
# 		age_model.eval()
# 		return gender_model, age_model
# 	elif model_type == 'cspvgg':
# 		from modules.vgg import CSP_VGG
# 		m = CSP_VGG(vgg_type='vgg19')
# 		m.load_state_dict(checkpoint['model_state_dict'])
# 		m.eval()
# 	elif model_type == 'inception':
# 		from modules.vgg import Inception_VGG
# 		m = Inception_VGG(vgg_type='vgg19')
# 		m.load_state_dict(checkpoint['model_state_dict'])
# 		m.eval()
# 		return m
#
# #####################################################################
#
# def detect_in_and_out(yolo_face, yolo_human, age_gender_model, args):
# 	assert isinstance(age_gender_model, tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
# 	gender_model, age_model = age_gender_model
#
# 	###### for human recognition #########
# 	# Get names and colors
# 	video_path = args.video
# 	# get stride and image size
# 	stride = int(yolo_human.stride.max())  # model stride
# 	img_size = check_img_size(640, s=stride)  # check img_size
# 	# Set Dataloader
# 	vid_path, vid_writer = None, None
# 	#### tracking objects
# 	face_tracker = CentroidTracker(maxDisappeared=60)
# 	person_tracker = CentroidTracker(maxDisappeared=60)
# 	### detected object ids
# 	saved_face_ids = []
# 	### person objects list
# 	faces = []
# 	### detected object ids
# 	saved_person_ids = []
# 	### person objects list
# 	persons = []
# 	# dataset = LoadStreams(video_path)
# 	expand_ratio = 0.0
# 	is_webcam = video_path.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
# 	if is_webcam:
# 		import torch.backends.cudnn as cudnn
# 		cudnn.benchmark = True
# 		dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
# 	else:
# 		dataset = LoadImages(video_path, img_size=img_size, stride=stride)
#
# 	# cap = VideoCapture(video_path)
#
# 	while True:
# 		if dataset is not None:
# 			for path, img, im0s, vid_cap in dataset:
# 				# while True:
# 				# 	img = cap.read()
# 				# 	im0s=[img]
# 				im0 = im0s[0] if is_webcam else im0s
# 				try:
# 					img = torch.from_numpy(img).to(device).float()
# 					img /= 255.0  # 0 - 255 to 0.0 - 1.0
# 					if img.ndimension() == 3:
# 						img = img.unsqueeze(0)
# 					with torch.no_grad():
# 						detected_person = yolo_human(img, augment=False)[0]
# 						detected_face = yolo_face(img, augment=False)[0]
# 					# Apply NMS
# 					detected_person = non_max_suppression(detected_person, args.person_score, args.iou, classes=0, agnostic=False)
# 					detected_face = non_max_suppression(detected_face, args.face_score, args.iou, classes=0, agnostic=False)
# 					tracking_faces = []
# 					tracking_persons = []
# 					# detected_objs = [torch.cat((detected_person[0], detected_face[0]))]
#
#
# 					for i, det in enumerate(detected_face):  # detections per image
# 						# Rescale boxes from img_size to im0 size
# 						det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.copy().shape).round()
#
# 						# 	# Write resultsqq
# 						gender, age = 'unknown','unknown'
# 						for *xyxy, conf, cls in reversed(det):
# 							if int(cls) == 0:
# 								#### tracking people in curent frame
#
# 								(x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
# 								tracking_faces.append([x1, y1, x2, y2])
# 								### expand bounding boxes by expand_ratio
# 								### x corresponding to column in numpy array --> dim = 1
# 								x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
# 								x2 = min(x2 + (expand_ratio * (x2 - x1)), im0.shape[1])
# 								### y corresponding to row in numpy array --> dim = 0
# 								y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
# 								y2 = min(y2 + (expand_ratio * (y2 - y1)), im0.shape[0])
# 								(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
# 								### extract the face
# 								det_img = im0[y1: y2, x1:x2,:].copy()
# 								# Predict Gender and Age
# 								with torch.no_grad():
# 									#### preparing face vector before passing to age, gender estimating model
# 									vectors = image_loader_(det_img)
# 									# age_gender_output = age_gender_model(vectors)
# 									pred_gender = gender_model(vectors)
# 									pred_age = age_model(vectors)
# 									## convert predicted index to meaningful label
# 									gender_indicate = pred_gender['gender'].argmax(dim=1).item()
# 									age_indicate = round(float(pred_age))
# 									gender = gender_choice.get(gender_indicate)
# 									age = age_choice.get(age_indicate)
# 									#print(f"detected! {gender} -- {age} years decade")
# 								if args.save_img or args.view_img:  # Add bbox to image
# 									cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 255), 2)
#
# 						### update tracking objects
# 						face_objects = face_tracker.update(tracking_faces)
# 						### object ids in current frame (reset each frame)
# 						current_object_ids = set()
# 						for (object_id, centroid) in face_objects.items():
# 							current_object_ids.add(object_id)
# 							if object_id not in saved_face_ids:
# 								if gender != 'unknown' and age != 'unknown':
# 									# print(f'there are new object: {object_id}')
# 									## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
# 									new_face = Face(id_=object_id, gender=[gender], age=[age], first_centroid=centroid)
# 									faces.append(new_face)
# 									saved_face_ids.append(object_id)
# 									# print(f'len(faces) {len(faces)}')
# 							else:
# 								if gender != 'unknown' and age != 'unknown':
# 									### when the face object is already in the managing face_objects, update it's info
# 									### get and edit
# 									old_face = faces[saved_face_ids.index(object_id)]
# 									old_face.gender = old_face.gender + [gender]
# 									old_face.age = old_face.age + [age]
# 									old_face.last_centroid = centroid
# 									### update
# 									faces[saved_face_ids.index(object_id)] = old_face
# 							#### draw rectangle bounding box for each face
# 							text = f"ID:{object_id}"
# 							# print(f'\n===============================')
# 							# print(text)
# 							# print(f'===============================\n')
# 							cv2.putText(img=im0,
# 										text=text,
# 										org=(centroid[0] - 10, centroid[1] - 10),
# 										fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# 										fontScale=0.55,
# 										color=(0, 255, 0),
# 										thickness=2)
# 							cv2.circle(img=im0,
# 									   center=(centroid[0], centroid[1]),
# 									   radius=4,
# 									   color=(0, 255, 0),
# 									   thickness=1)
# 						for obj in faces:
# 							if obj.id not in current_object_ids:  ### face disappeared
# 								gender = 'Male' if (obj.gender.count('male') >= obj.gender.count('female')) else 'Female'
# 								age = max(set(obj.age), key=obj.age.count)
# 								try:
# 									going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
# 									### remove disappeared object from face_objs and saved face_id
# 									if going_in:
# 										print(f'Someone is going in')
# 										try:
# 											send(obj.id, gender, age, going_in)
# 										except ConnectionError as e:
# 											print(f'sending data to database was failed. Check the database server connection..')
# 											continue
# 									faces.remove(obj)
# 									saved_face_ids.remove(obj.id)
#
# 								except Exception as e:
# 									if obj in faces:
# 										faces.remove(obj)
# 									if obj.id in saved_face_ids:
# 										saved_face_ids.remove(obj.id)
# 									continue
#
#
# 					for i, det in enumerate(detected_person):  # detections per image
# 						# Rescale boxes from img_size to im0 size
# 						det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.copy().shape).round()
#
# 						# 	# Write resultsqq
# 						for *xyxy, conf, cls in reversed(det):
# 							if int(cls) == 0:
# 								#### tracking people in curent frame
# 								(x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
# 								tracking_persons.append([x1, y1, x2, y2])
# 								### expand bounding boxes by expand_ratio
# 								### x corresponding to column in numpy array --> dim = 1
# 								x1 = max(x1 - (expand_ratio * (x2 - x1)), 0)
# 								x2 = min(x2 + (expand_ratio * (x2 - x1)), im0.shape[1])
# 								### y corresponding to row in numpy array --> dim = 0
# 								y1 = max(y1 - (expand_ratio * (y2 - y1)), 0)
# 								y2 = min(y2 + (expand_ratio * (y2 - y1)), im0.shape[0])
# 								(x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
# 								if args.save_img or args.view_img:  # Add bbox to image
# 									cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
# 						### update tracking objects
# 						person_objects = person_tracker.update(tracking_persons)
# 						### object ids in current frame (reset each frame)
# 						current_object_ids = set()
# 						for (object_id, centroid) in person_objects.items():
# 							current_object_ids.add(object_id)
# 							if object_id not in saved_person_ids:
# 								# print(f'there are new object: {object_id}')
# 								## when the person  object id is not in saved_person id. put the id into saved_object_id and put person object to person_objs for managing
# 								new_person = Person(id_=object_id, first_centroid=centroid)
# 								persons.append(new_person)
# 								saved_person_ids.append(object_id)
# 							else:
# 								### when the face object is already in the managing face_objects, update it's info
# 								### get and edit
# 								old_person = persons[saved_person_ids.index(object_id)]
# 								old_person.last_centroid = centroid
# 								### update
# 								persons[saved_person_ids.index(object_id)] = old_person
#
# 							#### draw rectangle bounding box for each person
# 							text = f"ID:{object_id}"
# 							# print(f'\n===============================')
# 							# print(text)
# 							# print(f'===============================\n')
# 							cv2.putText(img=im0,
# 										text=text,
# 										org=(centroid[0] - 10, centroid[1] - 10),
# 										fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# 										fontScale=0.55,
# 										color=(0, 255, 0),
# 										thickness=2)
# 							cv2.circle(img=im0,
# 									   center=(centroid[0], centroid[1]),
# 									   radius=4,
# 									   color=(0, 255, 0),
# 									   thickness=1)
# 						for obj in persons:
# 							if obj.id not in current_object_ids:  ### person disappeared
# 								### human recognition model does not have gender and age info
# 								gender = 'unknown'
# 								age = -1
# 								try:
# 									going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
# 									### remove disappeared object from (person_objs and saved person_id)
# 									if not going_in:
# 										print(f'Someone is going out')
# 										try:
# 											send(obj.id, gender, age, going_in)
# 										except ConnectionError as e:
# 											print(f'sending data to database was failed. Check the database server connection..')
# 											continue
# 									persons.remove(obj)
# 									saved_person_ids.remove(obj.id)
#
# 								except Exception as e:
# 									if obj in persons:
# 										persons.remove(obj)
# 									if obj.id in saved_person_ids:
# 										saved_person_ids.remove(obj.id)
# 									continue
# 						if args.view_img:
# 							cv2.imshow('view', im0)
# 					if cv2.waitKey(1) & 0xFF == ord('q'):
# 						break
# 				except NotImplementedError as e:
# 					print(f'Run_yolo.py 854. Error: {e}')
# 					continue
# 				if cv2.waitKey(1) & 0xFF == ord('q'):
# 					break
# 			break
# 		else:
# 			print('Dataset is None. Recreating..')
# 			dataset = LoadImages(video_path, img_size=img_size, stride=stride)
# 	cv2.destroyAllWindows()
#
# def get_args():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--model', type=str, default='models/checkpoints/yolov5s_face.pt', help='path to model weights file')
# 	parser.add_argument('--anchors', type=str, default='cfg/yolo_anchors.txt', help='path to anchor definitions')
# 	parser.add_argument('--classes', type=str, default='cfg/face_classes.txt', help='path to class definitions')
# 	parser.add_argument('--face-score', type=float, default=0.5, help='the face score threshold [0.0 ~ 1.0]')
# 	parser.add_argument('--person-score', type=float, default=0.45, help='the person score threshold [0.0 ~ 1.0]')
# 	parser.add_argument('--iou', type=float, default=0.45, help='the iou threshold [0.0 ~ 1.0]')
# 	parser.add_argument('--img-size', type=list, action='store', default=(416, 416), help='input image size')
# 	parser.add_argument('--image', default=False, action="store_true", help='image detection mode , boolean type')
# 	parser.add_argument('--video', type=str, default='rtsp://itaz:12345@192.168.0.33:554/stream_ch00_0', help='path to the video')
# 	parser.add_argument('--output', default=False, action="store_true", help='whether save the output to video file, boolean type')
# 	parser.add_argument('--weights', nargs='+', type=str, default='models/checkpoints/yolov5s.pt', help='yolo5 checkpoint path(s)')
# 	parser.add_argument('--view-img', action='store_true', help='display results, boolean type')
# 	parser.add_argument('--save-img', action='store_true', help='save results , boolean type')
# 	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt , boolean type')
# 	parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# 	args = parser.parse_args()
# 	return args
#
#
# if __name__ == "__main__":
# 	## load arguments
# 	args = get_args()
# 	### load models
# 	# yolov3 = YOLO(args)
#
# 	if isinstance(args.device, list):
# 		device = torch.device(f'cuda:{int(id_)}' for id_ in args.device)
# 	elif isinstance(args.device, int):
# 		device = torch.device(f'cuda:{int(args.device)}')
# 	else:
# 		device = torch.device('cpu')
# 	yolo_face = attempt_load(args.model, map_location=device)
# 	yolo_human = attempt_load(args.weights, map_location=device)
# 	age_gender_model = get_model()
# 	# detect_img(yolov3, r'G:\locs_projects\on_working\images\test_images', age_gender_model)
# 	# detect_img(yolov3, r'val_images', age_gender_model)
# 	detect_in_and_out(yolo_face, yolo_human, age_gender_model, args)
# ## run 2 models on separate threads
# # run_in = threading.Thread(target=detect_in, args=(yolov3, age_gender_model, args))
# # # run_out = threading.Thread(target=detect_out, args=(yolov5, args))
# #
# # # run_out.start()
# # run_in.start()
# # # detect_out(yolov5, args)
# # # run_out.join()
# # run_in.join()
