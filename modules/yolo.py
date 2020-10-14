# *******************************************************************
#
# Author : thien@locslab.com 2020
# with the reference of  : sthanhng@gmail.com
# on Github : https://github.com/sthanhng
# *******************************************************************
import colorsys
import datetime
import os

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


def detect_video(model, video_path=None, output=None, age_gender_model=None, original_object_points = None, target_object_points=None, transformer=None):
    ### do not change
    camera_focal_length = 2.2 ### do not change
    width_height_ratio = 640/480 ### do not change
    optical_centers_vertical = 0 ### do not change
    optical_centers_horizon = 450 ### do not change

    step = 0.0001
    transformer=np.array([[ camera_focal_length,  step*35000, optical_centers_vertical],
                         [0,  camera_focal_length * width_height_ratio, optical_centers_horizon],
                         [ 0,  step*45, 1.00000000e+00]])
    # transformer=np.array([[ 9.13897500e-01,  3.94517420e-01, -1.34498073e+02],
    #                      [ 4.04364735e-02,  1.14784958e+00, -7.26265561e+01],
    #                      [ 3.13679457e-04,  2.09795759e-03, 1.00000000e+00]])
    if transformer is None:
        src = np.float32([[original_object_points[0][0], original_object_points[8][0], original_object_points[45][0], original_object_points[53][0]]])
        dst = np.float32([target_object_points[0][0], target_object_points[8][0], target_object_points[45][0], target_object_points[53][0]])
        transformer = cv2.getPerspectiveTransform(src, dst)  # The transformation matrix



    if video_path == 'stream':
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # the video format and fps
    # video_fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    # the size of the frames to write
    video_size = (int(640),int(480))
    isOutput = True if output != "" else False
    out=None
    if isOutput:
        out = cv2.VideoWriter('outputs/output_video.mp4', video_fourcc, video_fps, video_size)

    tracker = CentroidTracker()
    offset = 200
    while True:
        tracking_faces = []
        ret, frames = vid.read()
        image = Image.fromarray(frames)
        image, faces = model.detect_image(image)
        gender = 'unknown'
        age = 'unknown'
        for i, (x1, y1, x2, y2) in enumerate(faces): ## with yolo, result will be 2 point of rectangle corner (x1, y1) and (x2, y2)
            ### each time detected a face, insert a new color
            (x1, y1, x2, y2)=(int(x1), int(y1), int(x2), int(y2))
            cv2.rectangle(frames, (y1, x1), (y2, x2), (255,0,0), 2)
            tracking_faces.append([y1, x1, y2, x2])
            ### extract the face
            face_img = frames[x1: x2, y1:y2].copy()
            # Predict Gender and Age
            try:
                #### preparing face vector before passing to age, gender estimating model
                vectors = image_loader_(face_img)
                age_gender_output = age_gender_model(vectors)
                ## convert predicted index to meaningful label
                sex_indicate = [i + 1 for i in age_gender_output['sex'].argmax(dim=-1).tolist()]
                age_indicate = [i + 1 for i in age_gender_output['age'].argmax(dim=-1).tolist()]
                # overlay_text = "%s, %s" % (sex_choice.get(sex_indicate[0]), age_choice.get(age_indicate[0]))
                # Draw a rectangle around the faces (we plot the resule, so, the text will not appear, plotting frames to see the text
                # cv2.putText(frames, overlay_text, (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                print(f"시점: {str(datetime.datetime.now().time())} -- 성별: {sex_choice.get(sex_indicate[0])} -- 나이: {age_choice.get(age_indicate[0])}\n")
                gender = sex_choice.get(sex_indicate[0])
                age = age_choice.get(age_indicate[0])
            except Exception as e:
                continue


        objects = tracker.update(tracking_faces)
        #### for map view, replace frames with zeros array:
        ### disable ths 2 lines
        map = cv2.imread('templates/map.png')
        map_view=False
        if not map_view:
            for (object_id, centroid) in objects.items():
                text = "ID {}, gender {}, age {}".format(object_id, gender, age)
                cv2.putText(frames, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frames, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            warped_img = cv2.warpPerspective(frames, transformer, (frames.shape[0]+offset*6, frames.shape[1]))  # Image warping
            ### crop and resize image
            start_point_vertical = np.max(np.where(warped_img[:, 0] > 0))  ### find the point at which the camera has image pixel
            width = np.max(np.where(warped_img[start_point_vertical] > 0))  ## find the width of camera image
            height = np.max(np.where(warped_img[:, int(width / 2), :] > 0))  ## find the height of camera image (include start_point

            crop_width_start_point = np.min(np.where((warped_img[:, height, :] > 0))[0])
            crop_width_end_point = np.max(np.where((warped_img[height] > 0))[0])

            # print(f'start_point_vertical {start_point_vertical}')
            # print(f'width {width}')
            # print(f'height {height}')
            # print(f'crop_width_start_point {crop_width_start_point}')
            # print(f'crop_width_end_point {crop_width_end_point}')

            cropped_warped_img = warped_img[start_point_vertical:start_point_vertical + height, crop_width_start_point:crop_width_end_point].copy()
            cropped_warped_img = cv2.resize(cropped_warped_img, (cropped_warped_img.shape[1] * 2, cropped_warped_img.shape[0] * 4))
            cv2.imshow("Map View", frames)

        else:
            empty_img = np.zeros_like(frames)
            for (object_id, centroid) in objects.items():
                text = "ID {}".format(object_id)
                cv2.putText(empty_img, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(empty_img, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            warped_img = cv2.warpPerspective(empty_img, transformer, (empty_img.shape[0] + offset * 6, empty_img.shape[1]))  # Image warping
            map = cv2.resize(map, (warped_img.shape[1], warped_img.shape[0]), interpolation=cv2.INTER_AREA)

            with_map_cropped_warped_img = warped_img + map
            cv2.imshow("Map View", with_map_cropped_warped_img)

        ### save video if needed
        if out is not None:
            out.write(frames)
        #### define interupt event
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()
    # close the session
    model.close_session()
    return gender, age

## backup
# # *******************************************************************
# import math
# import os, datetime
# import colorsys
# import numpy as np
# import cv2
# import torch
# from trac_object import createTrackerByName
# from modules.model import eval
#
# from keras import backend as K
# from keras.models import load_model
# from timeit import default_timer as timer
# from PIL import ImageDraw, Image
#
#
# class YOLO(object):
#     def __init__(self, args):
#         self.args = args
#         self.model_path = args.model
#         self.classes_path = args.classes
#         self.anchors_path = args.anchors
#         self.class_names = self._get_class()
#         self.anchors = self._get_anchors()
#         self.sess = K.get_session()
#         self.boxes, self.scores, self.classes = self._generate()
#         self.model_image_size = args.img_size
#
#     def _get_class(self):
#         classes_path = os.path.expanduser(self.classes_path)
#         with open(classes_path) as f:
#             class_names = f.readlines()
#         class_names = [c.strip() for c in class_names]
#         return class_names
#
#     def _get_anchors(self):
#         anchors_path = os.path.expanduser(self.anchors_path)
#         with open(anchors_path) as f:
#             anchors = f.readline()
#         anchors = [float(x) for x in anchors.split(',')]
#         return np.array(anchors).reshape(-1, 2)
#
#     def _generate(self):
#         model_path = os.path.expanduser(self.model_path)
#         assert model_path.endswith(
#             '.h5'), 'Keras model or weights must be a .h5 file'
#
#         # load model, or construct model and load weights
#         num_anchors = len(self.anchors)
#         num_classes = len(self.class_names)
#         try:
#             self.yolo_model = load_model(model_path, compile=False)
#         except:
#             # make sure model, anchors and classes match
#             self.yolo_model.load_weights(self.model_path)
#         else:
#             assert self.yolo_model.layers[-1].output_shape[-1] == \
#                    num_anchors / len(self.yolo_model.output) * (
#                            num_classes + 5), \
#                 'Mismatch between model and given anchor and class sizes'
#         print(
#             '*** {} model, anchors, and classes loaded.'.format(model_path))
#
#         # generate colors for drawing bounding boxes
#         hsv_tuples = [(x / len(self.class_names), 1., 1.)
#                       for x in range(len(self.class_names))]
#         self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
#         self.colors = list(
#             map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
#                 self.colors))
#
#         # shuffle colors to decorrelate adjacent classes.
#         np.random.seed(102)
#         np.random.shuffle(self.colors)
#         np.random.seed(None)
#
#         # generate output tensor targets for filtered bounding boxes.
#         self.input_image_shape = K.placeholder(shape=(2,))
#         boxes, scores, classes = eval(self.yolo_model.output, self.anchors,
#                                            len(self.class_names),
#                                            self.input_image_shape,
#                                            score_threshold=self.args.score,
#                                            iou_threshold=self.args.iou)
#         return boxes, scores, classes
#
#     def detect_image(self, image):
#         if self.model_image_size != (None, None):
#             assert self.model_image_size[
#                        0] % 32 == 0, 'Multiples of 32 required'
#             assert self.model_image_size[
#                        1] % 32 == 0, 'Multiples of 32 required'
#             boxed_image = letterbox_image(image, tuple(
#                 reversed(self.model_image_size)))
#         else:
#             new_image_size = (image.width - (image.width % 32),
#                               image.height - (image.height % 32))
#             boxed_image = letterbox_image(image, new_image_size)
#         image_data = np.array(boxed_image, dtype='float32')
#         image_data /= 255.
#         # add batch dimension
#         image_data = np.expand_dims(image_data, 0)
#         out_boxes, out_scores, out_classes = self.sess.run(
#             [self.boxes, self.scores, self.classes],
#             feed_dict={
#                 self.yolo_model.input: image_data,
#                 self.input_image_shape: [image.size[1], image.size[0]],
#                 K.learning_phase(): 0
#             })
#         thickness = (image.size[0] + image.size[1]) // 400
#
#         for i, c in reversed(list(enumerate(out_classes))):
#             box = out_boxes[i]
#             draw = ImageDraw.Draw(image)
#             top, left, bottom, right = box
#             top = max(0, np.floor(top + 0.5).astype('int32'))
#             left = max(0, np.floor(left + 0.5).astype('int32'))
#             bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
#             right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#             for thk in range(thickness):
#                 draw.rectangle(
#                     [left + thk, top + thk, right - thk, bottom - thk],
#                     outline=(51, 178, 255))
#             del draw
#         return image, out_boxes
#
#     def close_session(self):
#         self.sess.close()
#
#
# def letterbox_image(image, size):
#     '''Resize image with unchanged aspect ratio using padding'''
#
#     img_width, img_height = image.size
#     w, h = size
#     scale = min(w / img_width, h / img_height)
#     nw = int(img_width * scale)
#     nh = int(img_height * scale)
#
#     image = image.resize((nw, nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128, 128, 128))
#     new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
#     return new_image
#
#
# def detect_img(yolo):
#     while True:
#         img = input('*** Input image filename: ')
#         try:
#             image = Image.open(img)
#         except:
#             if img == 'q' or img == 'Q':
#                 break
#             else:
#                 print('*** Open Error! Try again!')
#                 continue
#         else:
#             res_image, _ = yolo.detect_image(image)
#             res_image.show()
#     yolo.close_session()
#
#
#
#
#
# def image_loader_(face_img):
#     from PIL import Image
#     from torchvision import transforms
#     import numpy as np
#     img_size = 64
#     image_set = []
#
#     preprocess = transforms.Compose([transforms.Resize(img_size),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                           std=[0.229, 0.224, 0.225])])
#     face_img = Image.fromarray(np.uint8(face_img))
#     img = preprocess(face_img)
#     image_set.append(img)
#     img = torch.stack(image_set, dim=0)
#     return img
#
#
# tracking_faces = {}
# life_time = 60
#
# def detect_video(model, video_path=None, output=None, age_gender_model=None):
#     sex_choice = {1: 'Male', 2: 'Female'}
#     age_choice = {1: '<10', 2: '10~19', 3: '20~29', 4: '30~39', 5: '40~49', 6: '50~59', 7: '60~69', 8: '70~79', 9: '80~89', 10: '>=90'}
#
#     if video_path == 'stream':
#         vid = cv2.VideoCapture(0)
#     else:
#         vid = cv2.VideoCapture(video_path)
#     if not vid.isOpened():
#         raise IOError("Couldn't open webcam or video")
#
#     # the video format and fps
#     # video_fourcc = int(vid.get(cv2.CAP_PROP_FOURCC))
#     video_fourcc = cv2.VideoWriter_fourcc('M', 'G', 'P', 'G')
#     video_fps = vid.get(cv2.CAP_PROP_FPS)
#
#     # the size of the frames to write
#     video_size = (int(640),
#                   int(480))
#     isOutput = True if output != "" else False
#     out=None
#     if isOutput:
#         output_fn = 'output_video.avi'
#         out = cv2.VideoWriter(os.path.join(output, output_fn), video_fourcc, video_fps, video_size)
#
#     tracker = createTrackerByName("CSRT")
#     # for i, newbox in enumerate(boxes):
#     #     color = (randint(0, 255), randint(0, 255), randint(0, 255))
#     #
#     #     p1 = (int(newbox[0]), int(newbox[1]))
#     #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
#     #     cv2.rectangle(frames, p1, p2, color, 2, 1)
#     from random import randint
#     # import time
#     colors=[(randint(0,255), randint(0,255), randint(0,255)) for _ in range(1000)]
#     while True:
#         ret, frames = vid.read()
#         if ret: ### if there are frames
#             image = Image.fromarray(frames)
#             image, faces = model.detect_image(image)
#             result = np.asarray(image)
#             for i, (x1, y1, x2, y2) in enumerate(faces): ## with yolo, result will be 2 point of rectangle corner (x1, y1) and (x2, y2)
#                 ### each time detected a face, insert a new color
#                 tracker.init(frames, (x1, y1, x2, y2))
#                 tracking_faces[len(tracking_faces)] = life_time
#                 success, boxes = tracker.update(frames)
#                 # check to see if the tracking was a success
#                 if success:
#                     cv2.rectangle(frames, (y1, x1), (y2, x2), colors[i], 2)
#
#                 x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.floor(x2), math.floor(y2) #### make sure they are integer (floor to make sure they are not bigger than image size)
#                 ### extend the face view 30% (to see widerly the face), if not needed, set extend to 0
#                 width_extend = 0#int((x2 - x1)*0.3)
#                 height_extend = 0#int((x2 - x1) * 0.3)
#                 if x1 - width_extend < 0 or x2 + width_extend > frames.shape[0]:
#                     width_extend=0
#                 if y1 - height_extend<0 or y2 + height_extend > frames.shape[1]:
#                     height_extend=0
#                 ### extract the face
#                 face_img = frames[x1-width_extend: x2+width_extend, y1-height_extend:y2+width_extend].copy()
#                 # Predict Gender and Age
#                 try:
#                     #### preparing face vector before passing to age, gender estimating model
#                     vectors = image_loader_(face_img)
#                     age_gender_output = age_gender_model(vectors)
#                     ## convert predicted index to meaningful label
#                     sex_indicate = [i + 1 for i in age_gender_output['sex'].argmax(dim=-1).tolist()]
#                     age_indicate = [i + 1 for i in age_gender_output['age'].argmax(dim=-1).tolist()]
#                     overlay_text = "%s, %s" % (sex_choice.get(sex_indicate[0]), age_choice.get(age_indicate[0]))
#                     # Draw a rectangle around the faces (we plot the resule, so, the text will not appear, plotting frames to see the text
#                     cv2.putText(frames, overlay_text, (y1, x1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#                     print(f"시점: {str(datetime.datetime.now().time())} -- 성별: {sex_choice.get(sex_indicate[0])} -- 나이: {age_choice.get(age_indicate[0])}\n")
#                 except Exception as e:
#                     print(e)
#                     continue
#                 # if int(time.time()) % 5 == 0:
#                 #     ### save face image for debugging
#                 #     img_path = f'templates\\{overlay_text}_{time.time()}.jpg'
#                 #     cv2.imwrite(img_path, face_img)
#
#             # objects = tracker.update(rects=detected_faces)
#             # for (obj_id, obj_centroid) in objects.items():
#             #     text = "ID {}".format(obj_id)
#             #     cv2.putText(result, text, (obj_centroid[0] - 10, obj_centroid[1] - 10),
#             #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             #     cv2.circle(result, (obj_centroid[0], obj_centroid[1]), 4, (0, 255, 0), -1)
#             cv2.namedWindow("face", cv2.WINDOW_NORMAL)
#
#             cv2.imshow("face", frames)
#             ### save video if needed
#             if out is not None:
#                 out.write(result)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else: ## if there are not any frames
#             break
#     vid.release()
#     cv2.destroyAllWindows()
#     # close the session
#     model.close_session()
#
#
#
#
#
#