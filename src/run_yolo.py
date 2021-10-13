

### thien.locslab@gmail.com
import argparse
import os
import random
import time
<<<<<<< HEAD

import numpy as np
import torch
from cv2 import cv2
from imutils.video import VideoStream

from modules.new_yolo import Face, Person, attempt_load
from utils.datasets import letterbox
=======
import uuid

import numpy as np
# import tensorflow as tf
import torch
from cv2 import cv2
from torch import nn

from modules.new_yolo import Face, Person, attempt_load, VideoCapture
from utils.datasets import letterbox
from utils.datasets import LoadStreams, LoadImages, letterbox

>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.trac_object import CentroidTracker

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
np.random.seed(1542)
<<<<<<< HEAD
=======
# tf.random.set_seed(1542)
# tf.compat.v1.disable_eager_execution()
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238

os.environ['KERAS_BACKEND'] = 'tensorflow'

gender_choice = {0: 'female', 1: 'male'}
age_choice = {0: '10', 1: '20', 2: '30', 3: '40', 4: '50'}

<<<<<<< HEAD
=======
#
# def image_loader_(face_img):
#     from PIL import Image
#     from torchvision import transforms
#     import numpy as np
#     from train_module.data_helper import img_size  #### img_size=64
#     # image_set = []
#     preprocess = transforms.Compose([transforms.Resize(img_size + 4),
#                                      transforms.RandomCrop(img_size),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      # transforms.Normalize(mean=mean,
#                                      #                      std=std)
#                                      ])
#     # face_img = Image.fromarray(np.uint8(face_img))
#     face_img = Image.fromarray(np.uint8(face_img))
#     # face_img = Image.open(face_img)
#     img = preprocess(face_img)
#     img = img.unsqueeze(dim=0)
#     return img


>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
def image_loader(face_img_np):
    from train_module.data_helper import image_preprocess
    img = image_preprocess(face_img_np, train=False)
    img = img.unsqueeze(0)
    return img


<<<<<<< HEAD
def send(id, gender, age, going_in, URL = 'http://192.168.20.9:8080/api'):
    import requests
    import json
=======
def send(id, gender, age, going_in):
    import requests
    import json
    URL = 'http://museum.locslab.com/api'
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
    headers = {
        'Content-Type': 'application/json',
    }
    str_ = 'mutation{createDetection(identifier:%d, age:\"%s\", gender:\"%s\", inAndOut:%s){identifier age gender inAndOut}}' % (
        id, age, gender, str(going_in).lower())
    data = dict(query=str_, variables={})
    print(data)
    requests.post(URL, data=json.dumps(data), headers=headers)


<<<<<<< HEAD
def get_model(device=torch.device('cpu')):
    ## load models weights
    age_model_path = 'models/checkpoints/age-best.pth'
    gender_model_path = 'models/checkpoints/gender-best.pth'
    age_checkpoint = torch.load(age_model_path, map_location=device)
    gender_checkpoint = torch.load(gender_model_path, map_location=device)
=======

# torch.save({
# 			'parameter': {
# 				'epoch': epochs,
# 				'iterator': it,
# 				'batch_size': self.arguments.batch_size,
# 				'learning_rate': self.arguments.learning_rate
# 			},
# 			'model_weights': self.model.state_dict(),
# 			'optimizer_weights': self.optimizer.state_dict(),
# 			'model_type': self.arguments.model,
# 			'type': self.type ### gender or age model?
# 		}, filepath)
### get pretrained model of age and gender detection
def get_model(device=torch.device('cpu')):
    age_model_path = 'models/checkpoints/age-best.pth'
    gender_model_path = 'models/checkpoints/gender-best.pth'
    # gender_model_path = 'models/vgg19-epochs_97-step_0-gender_accuracy_0.979676459052346.pth'
    age_checkpoint = torch.load(age_model_path, map_location=device)
    gender_checkpoint = torch.load(gender_model_path, map_location=device)
    #
    weights = {}
    # for k, v in checkpoint_2['model_state_dict'].items():
    #     k = k.replace('sex', 'gender')
    #     weights[k] = v
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238

    from modules.vgg import Gender_VGG, Age_VGG
    gender_model = Gender_VGG(vgg_type='vgg19').to(device)
    age_model = Age_VGG(vgg_type='vgg19').to(device)
    gender_model.load_state_dict(gender_checkpoint['model_weights'])
    age_model.load_state_dict(age_checkpoint['model_weights'])

    gender_model.eval()
    age_model.eval()
    return gender_model, age_model


#####################################################################
@torch.no_grad()
<<<<<<< HEAD
def detect_in_and_out_(yolo_face, yolo_human, age_gender_model, args):
=======
def detect_in_and_out(yolo_face, yolo_human, age_gender_model, args):
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
    start = time.time()
    assert isinstance(age_gender_model,
                      tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
    gender_model, age_model = age_gender_model

<<<<<<< HEAD
    video_path = args.video.split(',')
    # get stride and image size
    stride = int(yolo_human.stride.max())  # model stride
    img_size = check_img_size(320, s=stride)  # check img_size
    #### tracking objects
    face_trackers = [CentroidTracker(maxDisappeared=40) for _ in range(6)]
    person_trackers = [CentroidTracker(maxDisappeared=40) for _ in range(6)]
    ### detected object ids
    saved_face_ids = [[] for _ in range(6)]
    ### person objects list
    faces = [[] for _ in range(6)]
    ### detected object ids
    saved_person_ids = [[] for _ in range(6)]
    ### person objects list
    persons = [[] for _ in range(6)]
    expand_ratio = [0.05,0.0]
    print(f'')

    print(f'====== Loading camera ========')
    cams = [VideoStream(path) for path in video_path]
    streams = [cams[i].start() for i in range(len(video_path))]
=======
    ###### for human recognition #########
    # Get names and colors
    video_path = args.video
    # get stride and image size
    stride = int(yolo_human.stride.max())  # model stride
    # stride = int(16)  # model stride
    img_size = check_img_size(640, s=stride)  # check img_size
    #### tracking objects
    face_tracker = CentroidTracker(maxDisappeared=20)
    person_tracker = CentroidTracker(maxDisappeared=20)
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
    # is_webcam = video_path.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    # if is_webcam:
    # 	import torch.backends.cudnn as cudnn
    # 	cudnn.benchmark = True
    # 	dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
    # else:
    # 	dataset = LoadImages(video_path, img_size=img_size, stride=stride)
    print(f'')

    print(f'====== Loading camera ========')
    cap = VideoCapture(video_path)
    vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get_fps()/5, (cap.get_size()))
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
    print(f'==== Camera loading takes {time.time()-start} seconds ====')
    print(f'Running...')
    print(f'')

<<<<<<< HEAD
    keep_rate=40
    result={}
    remove_key = []
    id_=0

    while True:
        ### load frames from all cameras
        all_frames = [streams[i].read() for i in range(len(streams))]

        ### loop over each camera and process separately
        for jj,frames in enumerate(all_frames):
            print(f'detecting camera: {jj}')
            if frames is None:
                ### if camera has error (ussally because of machine's latency), recreate it
                print(f'camera: {jj} has empty frame..')
                cams[jj] = VideoStream(video_path[jj])
                streams[jj] = cams[jj].start()
                print(f'Recreated..')
            else:
                frames = cv2.resize(src=frames, dsize=(640, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                frames = cv2.rotate(frames, cv2.ROTATE_90_CLOCKWISE)
                cv2.putText(img=frames,
                            text='Credit: locslab.com',
                            org=(10, 20),
=======
    criterion = nn.MSELoss()
    keep_rate=10
    result={}
    remove_key = []
    id_=0
    while True:

        id_+=1
        frames = cap.read()
        # cv2.putText(img=frames,
        #             text='Credit: thien.tdplaza@gmail.com',
        #             org=(10, 20),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.55,
        #             color=(25, 25, 255),
        #             thickness=1)

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
            curr_gender, curr_age = 'unknown', 'unknown'
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
                    # _id = uuid.uuid1().int
                    # cv2.imwrite(f'dataset/test/{_id}.png', det_img)
                    try:
                        # Predict Gender and Age
                        with torch.no_grad():
                            #### preparing face vector before passing to age, gender estimating model
                            vectors = image_loader(det_img).to(device)
                            # age_gender_output = age_gender_model(vectors)
                            pred_gender = gender_model(vectors)
                            # male_score = criterion(pred_gender, torch.tensor([0,1],dtype=torch.float))
                            # female_score = criterion(pred_gender, torch.tensor([1,0],dtype=torch.float))
                            # print(f'')
                            # print(f'pred_gender {pred_gender}')
                            # print(f'male_score {male_score}')
                            # print(f'female_score {female_score}')
                            pred_age = age_model(vectors)
                            ## convert predicted index to meaningful label
                            gender_indicate = pred_gender.argmax(dim=1).item()
                            age_indicate = round(float(pred_age))
                            curr_gender = gender_choice.get(gender_indicate)
                            curr_age = age_choice.get(age_indicate)
                            print(f'detected: gender {curr_gender} -- age {curr_age}')
                    except RuntimeError as e:
                        print('run_yolo 227')
                        continue
                    # print(f"detected! {gender} -- {age} years decade")
                    if args.save_img or args.view_img:  # Add bbox to image
                        text = f"{curr_gender}-{curr_age}"
                        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        cv2.putText(img=frames,
                                    text=text,
                                    org=(x1, y1),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.55,
                                    color=(255, 0, 255),
                                    thickness=1)
                        # cv2.imwrite(filename=f'{id_}.jpg', img=det_img)
            ### update tracking objects
            face_objects = face_tracker.update(tracking_faces)
            ### object ids in current frame (reset each frame)
            current_object_ids = set()
            for (object_id, centroid) in face_objects.items():
                current_object_ids.add(object_id)
                if object_id not in saved_face_ids:
                    if curr_gender != 'unknown' and curr_age != 'unknown':
                        # print(f'there are new object: {object_id}')
                        ## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
                        new_face = Face(id_=object_id, gender=[curr_gender], age=[curr_age], first_centroid=centroid)
                        faces.append(new_face)
                        saved_face_ids.append(object_id)
                    # print(f'len(faces) {len(faces)}')
                else:
                    if curr_gender != 'unknown' and curr_age != 'unknown':
                        ### when the face object is already in the managing face_objects, update it's info
                        ### get and edit
                        old_face = faces[saved_face_ids.index(object_id)]
                        old_face.gender = old_face.gender + [curr_gender]
                        old_face.age = old_face.age + [curr_age]
                        old_face.last_centroid = centroid
                        ### update
                        faces[saved_face_ids.index(object_id)] = old_face
                #### draw rectangle bounding box for each face
                # gender_ = 'Male' if (obj_.gender.count('male') >= obj_.gender.count('female')) else 'Female'
                # age_ = max(set(obj_.age), key=obj_.age.count)
                text = f"{object_id}"
                # print(f'\n===============================')
                # print(text)
                # print(f'===============================\n')
                cv2.putText(img=frames,
                            text=text,
                            org=(centroid[0] - 10, centroid[1] - 10),
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(255, 0, 255),
                            thickness=1)
<<<<<<< HEAD
                ## process frame image, to match yolov5 requirements
                img = [letterbox(x, img_size, auto=True, stride=stride)[0] for x in [frames]]
                img = np.stack(img, axis=0)
                # Convert img
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # ToTensor image convert to BGR, RGB
                img = np.ascontiguousarray(img)  # frame imgs, 메모리에 연속 배열 (ndim> = 1)을 반환
                img = torch.from_numpy(img).to(device)  # numpy array convert
                img = img.float()  # unit8 to 16/32
                img /= 255.0  # 0~255 to 0.0~1.0 images.
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                #### detecting
                detected_person = yolo_human(img, augment=False)[0]
                detected_face = yolo_face(img, augment=False)[0]
                # Apply NMS
                detected_person = non_max_suppression(detected_person, args.person_score, args.iou, classes=0, agnostic=False)
                detected_face = non_max_suppression(detected_face, args.face_score, args.iou, classes=0, agnostic=False)
                tracking_faces = []
                tracking_persons = []
                for i, det in enumerate(detected_face):  # detections per image
                    # Rescale boxes from img_size to frames size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()
                    # 	# Write result
                    curr_gender, curr_age = 'unknown', 'unknown'
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:
                            #### tracking people in curent frame
                            (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                            tracking_faces.append([x1, y1, x2, y2])
                            ### expand bounding boxes by expand_ratio
                            ### x corresponding to column in numpy array --> dim = 1
                            x1 = max(x1 - (expand_ratio[0] * (x2 - x1)), 0)
                            x2 = min(x2 + (expand_ratio[0] * (x2 - x1)), frames.shape[1])
                            ### y corresponding to row in numpy array --> dim = 0
                            y1 = max(y1 - (expand_ratio[1] * (y2 - y1)), 0)
                            y2 = min(y2 + (expand_ratio[1] * (y2 - y1)), frames.shape[0])
                            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                            ### extract the face
                            det_img = frames[y1: y2, x1:x2, :].copy()
                            try:
                                # Predict Gender and Age
                                with torch.no_grad():
                                    #### preparing face vector before passing to age, gender estimating model
                                    vectors = image_loader(det_img).to(device)
                                    pred_gender = gender_model(vectors)
                                    pred_age = age_model(vectors)
                                    ## convert predicted index to meaningful label
                                    gender_indicate = pred_gender.argmax(dim=1).item()
                                    age_indicate = round(float(pred_age))
                                    curr_gender = gender_choice.get(gender_indicate)
                                    curr_age = age_choice.get(age_indicate)
                                    print(f'detected on camera {jj}: gender {curr_gender} -- age {curr_age}')
                            except Exception as e:
                                print('run_yolo 595')
                                continue
                            if args.save_img or args.view_img:  # Add bbox to image
                                text = f"{curr_gender}-{curr_age}"
                                cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 255), 1)
                                cv2.putText(img=frames,
                                            text=text,
                                            org=(x1, y1),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                            fontScale=0.55,
                                            color=(255, 0, 255),
                                            thickness=1)
                    ### update tracking objects
                    face_objects = face_trackers[jj].update(tracking_faces)
                    ### object ids in current frame (reset each frame)
                    current_object_ids = set()
                    for (object_id, centroid) in face_objects.items():
                        current_object_ids.add(object_id)
                        if object_id not in saved_face_ids[jj]:
                            if curr_gender != 'unknown' and curr_age != 'unknown':
                                ## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
                                new_face = Face(id_=object_id, gender=[curr_gender], age=[curr_age], first_centroid=centroid)
                                faces[jj].append(new_face)
                                saved_face_ids[jj].append(object_id)
                        else:
                            if curr_gender != 'unknown' and curr_age != 'unknown':
                                ### when the face object is already in the managing face_objects, update it's info
                                old_face = faces[jj][saved_face_ids[jj].index(object_id)]
                                old_face.gender = old_face.gender + [curr_gender]
                                old_face.age = old_face.age + [curr_age]
                                old_face.last_centroid = centroid
                                ### update
                                faces[jj][saved_face_ids[jj].index(object_id)] = old_face
                        text = f"{object_id}"
                        cv2.putText(img=frames,
                                    text=text,
                                    org=(centroid[0] - 10, centroid[1] - 10),
=======
                cv2.circle(img=frames,
                           center=(centroid[0], centroid[1]),
                           radius=4,
                           color=(255, 0, 255),
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
                                txt = f'id:{obj.id}, gender: {gender}, age: {age} has gone in'
                                result[txt] = keep_rate
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
            p_expand_ratio=0.0
            # 	# Write resultsqq
            for *xyxy, conf, cls in reversed(det):
                if int(cls) == 0:
                    #### tracking people in curent frame
                    (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    tracking_persons.append([x1, y1, x2, y2])
                    ### expand bounding boxes by p_expand_ratio
                    ### x corresponding to column in numpy array --> dim = 1
                    x1 = max(x1 - (p_expand_ratio * (x2 - x1)), 0)
                    x2 = min(x2 + (p_expand_ratio * (x2 - x1)), frames.shape[1])
                    ### y corresponding to row in numpy array --> dim = 0
                    y1 = max(y1 - (p_expand_ratio * (y2 - y1)), 0)
                    y2 = min(y2 + (p_expand_ratio * (y2 - y1)), frames.shape[0])
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    if args.save_img or args.view_img:  # Add bbox to image
                        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 255, 0), 2)

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
                text = f"{object_id}"
                # print(f'\n===============================')
                # print(text)
                # print(f'===============================\n')
                cv2.putText(img=frames,
                            text=text,
                            org=(centroid[0] - 10, centroid[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(255, 255, 0),
                            thickness=1)
                cv2.circle(img=frames,
                           center=(centroid[0], centroid[1]),
                           radius=4,
                           color=(255, 255, 0),
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
                                txt = f'id:{obj.id} has gone out'
                                result[txt] = keep_rate
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

            # cv2.rectangle(frames, (0,10), (320,240), (255, 0, 0), 2)
            for i,r in enumerate(result.keys()):
                cv2.putText(img=frames,
                            text=r,
                            org=(10, 20 * (i+1)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(0, 255, 0),
                            thickness=2,
                            bottomLeftOrigin=False)
                result[r] = result[r] - 1
                if result[r] == 0:
                    remove_key.append(r)

        for key in remove_key:
            result.pop(key)
        remove_key=[]

        if args.view_img:
            cv2.imshow('view', frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if args.save_img:
            vid_writer.write(frames)

    vid_writer.release()
    cv2.destroyAllWindows()

#####################################################################
@torch.no_grad()
def detect_in_and_out_(yolo_face, yolo_human, age_gender_model, args):
    start = time.time()
    assert isinstance(age_gender_model,
                      tuple), 'run_yolo.py line 440. please pass gender and age model as a tuble to detect'
    gender_model, age_model = age_gender_model

    ###### for human recognition #########
    # Get names and colors
    video_path = args.video
    # get stride and image size
    stride = int(yolo_human.stride.max())  # model stride
    # stride = int(16)  # model stride
    img_size = check_img_size(640, s=stride)  # check img_size
    #### tracking objects
    face_tracker = CentroidTracker(maxDisappeared=40)
    person_tracker = CentroidTracker(maxDisappeared=40)
    ### detected object ids
    saved_face_ids = []
    ### person objects list
    faces = []
    ### detected object ids
    saved_person_ids = []
    ### person objects list
    persons = []
    # dataset = LoadStreams(video_path)
    expand_ratio = [0.05,0.0]
    # is_webcam = video_path.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    # if is_webcam:
    # 	import torch.backends.cudnn as cudnn
    # 	cudnn.benchmark = True
    # 	dataset = LoadStreams(video_path, img_size=img_size, stride=stride)
    # else:
    # 	dataset = LoadImages(video_path, img_size=img_size, stride=stride)
    print(f'')

    print(f'====== Loading camera ========')
    # cap = cv2.VideoCapture(video_path)
    cap = VideoCapture(video_path)
    FPS = int(cap.cap.get(5))

    width = 640 if int(cap.cap.get(3))>640 else int(cap.cap.get(3))
    height = 480 if int(cap.cap.get(4))>480 else int(cap.cap.get(4))
    if args.save_img:
        vid_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MPEG'), FPS, (width,height))
    print(f'==== Camera loading takes {time.time()-start} seconds ====')
    print(f'Running...')
    print(f'')

    keep_rate=40
    result={}
    remove_key = []
    id_=0
    ### use this to read video frame by frame (to keep the video frame rate as original)
    # detected_video = []
    # video_frames=[]
    # for i in range(int(cap.get(7)/3)):
    #     ret, frame = cap.read()
    #     if ret:
    #         video_frames.append(frame)

    while True:
        frames = cap.read()
        frames = cv2.resize(src=frames, dsize=(width,height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.putText(img=frames,
                    text='Credit: locslab.com',
                    org=(10, 20),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.55,
                    color=(255, 0, 255),
                    thickness=1)


        id_+=1
        # print(f'{id_}/{int(cap.cap.get(7)/3)}')
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
            curr_gender, curr_age = 'unknown', 'unknown'
            for *xyxy, conf, cls in reversed(det):
                if int(cls) == 0:
                    #### tracking people in curent frame
                    # (x1, y1, x2, y2) = (int(xyxy[0]) * 2, int(xyxy[1]) * 2, int(xyxy[2]) * 2, int(xyxy[3]) * 2)
                    (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    tracking_faces.append([x1, y1, x2, y2])
                    ### expand bounding boxes by expand_ratio
                    ### x corresponding to column in numpy array --> dim = 1
                    x1 = max(x1 - (expand_ratio[0] * (x2 - x1)), 0)
                    x2 = min(x2 + (expand_ratio[0] * (x2 - x1)), frames.shape[1])
                    ### y corresponding to row in numpy array --> dim = 0
                    y1 = max(y1 - (expand_ratio[1] * (y2 - y1)), 0)
                    y2 = min(y2 + (expand_ratio[1] * (y2 - y1)), frames.shape[0])
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    ### extract the face
                    det_img = frames[y1: y2, x1:x2, :].copy()
                    # _id = uuid.uuid1().int
                    # cv2.imwrite(f'dataset/test/{_id}.png', det_img)
                    try:
                        # Predict Gender and Age
                        with torch.no_grad():
                            #### preparing face vector before passing to age, gender estimating model
                            vectors = image_loader(det_img).to(device)
                            # age_gender_output = age_gender_model(vectors)
                            pred_gender = gender_model(vectors)
                            # male_score = criterion(pred_gender, torch.tensor([0,1],dtype=torch.float))
                            # female_score = criterion(pred_gender, torch.tensor([1,0],dtype=torch.float))
                            # print(f'')
                            # print(f'pred_gender {pred_gender}')
                            # print(f'male_score {male_score}')
                            # print(f'female_score {female_score}')
                            pred_age = age_model(vectors)
                            ## convert predicted index to meaningful label
                            gender_indicate = pred_gender.argmax(dim=1).item()
                            age_indicate = round(float(pred_age))
                            curr_gender = gender_choice.get(gender_indicate)
                            curr_age = age_choice.get(age_indicate)
                            # print(f'detected: gender {curr_gender} -- age {curr_age}')
                    except RuntimeError as e:
                        print('run_yolo 595')
                        continue
                    # print(f"detected! {gender} -- {age} years decade")
                    if args.save_img or args.view_img:  # Add bbox to image
                        text = f"{curr_gender}-{curr_age}"
                        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        cv2.putText(img=frames,
                                    text=text,
                                    org=(x1, y1),
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.55,
                                    color=(255, 0, 255),
                                    thickness=1)
<<<<<<< HEAD
                        cv2.circle(img=frames,
                                   center=(centroid[0], centroid[1]),
                                   radius=4,
                                   color=(255, 0, 255),
                                   thickness=1)
                    ### check current detected faces, if it is disapered, determine that it is going in or out and count
                    for obj in faces[jj]:
                        if obj.id not in current_object_ids:  ### face disappeared
                            gender = 'Male' if (obj.gender.count('male') >= obj.gender.count('female')) else 'Female'
                            age = max(set(obj.age), key=obj.age.count)
                            try:
                                going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
                                ### remove disappeared object from face_objs and saved face_id
                                if going_in:
                                    print(f'Someone is going in on camera {jj}')
                                    try:
                                        txt = f'id:{obj.id}, gender: {gender}, age: {age} has gone in'
                                        result[txt] = keep_rate
                                        send(obj.id, gender, age, going_in)
                                    except ConnectionError as e:
                                        print(
                                            f'sending data to database was failed. Check the database server connection..')
                                        continue
                                faces[jj].remove(obj)
                                saved_face_ids[jj].remove(obj.id)
                            except Exception as e:
                                print('')
                                print(e)
                                if obj in faces[jj]:
                                    faces[jj].remove(obj)
                                if obj.id in saved_face_ids[jj]:
                                    saved_face_ids[jj].remove(obj.id)
                                continue
                for i, det in enumerate(detected_person):  # detections per image
                    # Rescale boxes from img_size to frames size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()
                    p_expand_ratio=0.0
                    # 	# Write resultsqq
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:
                            #### tracking people in curent frame
                            (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                            tracking_persons.append([x1, y1, x2, y2])
                            ### expand bounding boxes by p_expand_ratio
                            ### x corresponding to column in numpy array --> dim = 1
                            x1 = max(x1 - (p_expand_ratio * (x2 - x1)), 0)
                            x2 = min(x2 + (p_expand_ratio * (x2 - x1)), frames.shape[1])
                            ### y corresponding to row in numpy array --> dim = 0
                            y1 = max(y1 - (p_expand_ratio * (y2 - y1)), 0)
                            y2 = min(y2 + (p_expand_ratio * (y2 - y1)), frames.shape[0])
                            (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                            if args.save_img or args.view_img:  # Add bbox to image
                                cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    ### update tracking objects
                    person_objects = person_trackers[jj].update(tracking_persons)
                    ### object ids in current frame (reset each frame)
                    current_object_ids = set()
                    for (object_id, centroid) in person_objects.items():
                        current_object_ids.add(object_id)
                        if object_id not in saved_person_ids[jj]:
                            ## when the person  object id is not in saved_person id. put the id into saved_object_id and put person object to person_objs for managing
                            new_person = Person(id_=object_id, first_centroid=centroid)
                            persons[jj].append(new_person)
                            saved_person_ids[jj].append(object_id)
                        else:
                            ### when the face object is already in the managing face_objects, update it's info
                            ### get and edit
                            old_person = persons[jj][saved_person_ids[jj].index(object_id)]
                            old_person.last_centroid = centroid
                            ### update
                            persons[jj][saved_person_ids[jj].index(object_id)] = old_person

                        #### draw rectangle bounding box for each person
                        text = f"{object_id}"
                        cv2.putText(img=frames,
                                    text=text,
                                    org=(centroid[0] - 10, centroid[1] - 10),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.55,
                                    color=(255, 255, 0),
                                    thickness=1)
                        cv2.circle(img=frames,
                                   center=(centroid[0], centroid[1]),
                                   radius=4,
                                   color=(255, 255, 0),
                                   thickness=1)
                    for obj in persons[jj]:
                        if obj.id not in current_object_ids:  ### person disappeared
                            ### human recognition model does not have gender and age info
                            gender = 'unknown'
                            age = -1
                            try:
                                going_in = True if obj.first_centroid[-1] < obj.last_centroid[-1] else False
                                ### remove disappeared object from (person_objs and saved person_id)
                                if not going_in:
                                    print(f'Someone is going out on camera {jj}')
                                    try:
                                        txt = f'id:{obj.id} has gone out'
                                        result[txt] = keep_rate
                                        send(obj.id, gender, age, going_in)
                                    except ConnectionError as e:
                                        print(
                                            f'sending data to database was failed. Check the database server connection..')
                                        continue
                                persons[jj].remove(obj)
                                saved_person_ids[jj].remove(obj.id)
                            except Exception as e:
                                print('')
                                #print(video_path[jj])
                                print(e)
                                if obj in persons[jj]:
                                    persons[jj].remove(obj)
                                if obj.id in saved_person_ids[jj]:
                                    saved_person_ids[jj].remove(obj.id)
                                continue
                ### this just for display info on frame, if we do not need to see the frames, this is useless
                if args.save_img and jj==0:
                    frames = cv2.rotate(frames, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    for i,r in enumerate(result.keys()):
                        cv2.putText(img=frames,
                                    text=r,
                                    org=(10, 40 * (i+1)),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.55,
                                    color=(0, 255, 0),
                                    thickness=2,
                                    bottomLeftOrigin=False)
                        result[r] = result[r] - 1
                        if result[r] == 0:
                            remove_key.append(r)
                    for key in remove_key:
                        result.pop(key)
                    remove_key=[]

=======
                        # cv2.imwrite(filename=f'{id_}.jpg', img=det_img)
            ### update tracking objects
            face_objects = face_tracker.update(tracking_faces)
            ### object ids in current frame (reset each frame)
            current_object_ids = set()
            for (object_id, centroid) in face_objects.items():
                current_object_ids.add(object_id)
                if object_id not in saved_face_ids:
                    if curr_gender != 'unknown' and curr_age != 'unknown':
                        # print(f'there are new object: {object_id}')
                        ## when the face  object id is not in saved_face id. put the id into saved_object_id and put face object to face_objs for managing
                        new_face = Face(id_=object_id, gender=[curr_gender], age=[curr_age], first_centroid=centroid)
                        faces.append(new_face)
                        saved_face_ids.append(object_id)
                    # print(f'len(faces) {len(faces)}')
                else:
                    if curr_gender != 'unknown' and curr_age != 'unknown':
                        ### when the face object is already in the managing face_objects, update it's info
                        ### get and edit
                        old_face = faces[saved_face_ids.index(object_id)]
                        old_face.gender = old_face.gender + [curr_gender]
                        old_face.age = old_face.age + [curr_age]
                        old_face.last_centroid = centroid
                        ### update
                        faces[saved_face_ids.index(object_id)] = old_face
                #### draw rectangle bounding box for each face
                # gender_ = 'Male' if (obj_.gender.count('male') >= obj_.gender.count('female')) else 'Female'
                # age_ = max(set(obj_.age), key=obj_.age.count)
                text = f"{object_id}"
                # print(f'\n===============================')
                # print(text)
                # print(f'===============================\n')
                cv2.putText(img=frames,
                            text=text,
                            org=(centroid[0] - 10, centroid[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(255, 0, 255),
                            thickness=1)
                cv2.circle(img=frames,
                           center=(centroid[0], centroid[1]),
                           radius=4,
                           color=(255, 0, 255),
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
                                txt = f'id:{obj.id}, gender: {gender}, age: {age} has gone in'
                                result[txt] = keep_rate
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
            p_expand_ratio=0.0
            # 	# Write resultsqq
            for *xyxy, conf, cls in reversed(det):
                if int(cls) == 0:
                    #### tracking people in curent frame
                    (x1, y1, x2, y2) = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    tracking_persons.append([x1, y1, x2, y2])
                    ### expand bounding boxes by p_expand_ratio
                    ### x corresponding to column in numpy array --> dim = 1
                    x1 = max(x1 - (p_expand_ratio * (x2 - x1)), 0)
                    x2 = min(x2 + (p_expand_ratio * (x2 - x1)), frames.shape[1])
                    ### y corresponding to row in numpy array --> dim = 0
                    y1 = max(y1 - (p_expand_ratio * (y2 - y1)), 0)
                    y2 = min(y2 + (p_expand_ratio * (y2 - y1)), frames.shape[0])
                    (x1, y1, x2, y2) = (int(x1), int(y1), int(x2), int(y2))
                    if args.save_img or args.view_img:  # Add bbox to image
                        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 255, 0), 2)

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
                text = f"{object_id}"
                # print(f'\n===============================')
                # print(text)
                # print(f'===============================\n')
                cv2.putText(img=frames,
                            text=text,
                            org=(centroid[0] - 10, centroid[1] - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(255, 255, 0),
                            thickness=1)
                cv2.circle(img=frames,
                           center=(centroid[0], centroid[1]),
                           radius=4,
                           color=(255, 255, 0),
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
                                txt = f'id:{obj.id} has gone out'
                                result[txt] = keep_rate
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
        if args.save_img:
            # cv2.rectangle(frames, (0,10), (320,240), (255, 0, 0), 2)
            for i,r in enumerate(result.keys()):
                cv2.putText(img=frames,
                            text=r,
                            org=(10, 40 * (i+1)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.55,
                            color=(0, 255, 0),
                            thickness=2,
                            bottomLeftOrigin=False)
                result[r] = result[r] - 1
                if result[r] == 0:
                    remove_key.append(r)

            for key in remove_key:
                result.pop(key)
            remove_key=[]
            vid_writer.write(frames)
        if args.view_img:
            cv2.imshow('view', frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if args.save_img:
        vid_writer.release()
    cv2.destroyAllWindows()
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238


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
<<<<<<< HEAD
    parser.add_argument('--video', type=str, default=None, help='path to the video')
=======
    parser.add_argument('--video', type=str, default='rtsp://itaz:12345@192.168.0.33:554/stream_ch00_0',
                        help='path to the video')
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
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
    print('')
    print('=============== Loading models ==============')
    start = time.time()
    ## load arguments
    args = get_args()
    ### load models
    # yolov3 = YOLO(args)
    if isinstance(args.device, int):
        device = torch.device(f'cuda:{int(args.device)}') if torch.cuda.is_available() else torch.device('cpu')
    elif isinstance(args.device, list):
        device = [torch.device(f'cuda:{int(d)}') for d in args.device]
    else:
        device = torch.device('cpu')
    yolo_face = attempt_load(args.model, map_location=device)
    yolo_human = attempt_load(args.weights, map_location=device)
    age_gender_model = get_model(device=device)
    print(f'=============== models loaded, it takes {time.time()-start} seconds ==============')
<<<<<<< HEAD
=======

    # detect_img(yolov3, r'G:\locs_projects\on_working\images\test_images', age_gender_model)
    # detect_img(yolov3, r'val_images', age_gender_model)
>>>>>>> be0df902260345b48441f8c6dcc4ccc975d1e238
    detect_in_and_out_(yolo_face, yolo_human, age_gender_model, args)