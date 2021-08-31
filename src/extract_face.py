### thien.locslab@gmail.com
import argparse
import os

import numpy as np
# import tensorflow as tf
import torch
from cv2 import cv2

from modules.new_yolo import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/checkpoints/yolov5s_face.pt',
                        help='path to model weights file')
    parser.add_argument('--path', type=str, default=os.path.join('dataset', 'train_image_unbalanced'), help='path to img folder')
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


def extract(model, path):
    # path = r'dataset/test_images'
    model.eval()
    stride = int(32)  # model stride
    img_size = check_img_size(64, s=stride)
    with torch.no_grad():
        for file in os.listdir(path=path):
            img_path=os.path.join(path,file)
            frames = cv2.imread(img_path)
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
            print(img.shape)
            detected_face = model(img, augment=False)[0]
            # detected_face = non_max_suppression(detected_face, 0.6, 0.45, classes=0, agnostic=False)
            expand_ratio=0.05
            # save_path = img_path.replace(img_path.split('/')[-2],'extracted_faces')
            save_path = img_path.replace('train_image_unbalanced','extracted_faces')
            print(save_path)

            for det in detected_face:  # detections per image
                # Rescale boxes from img_size to frames size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frames.copy().shape).round()

                # 	# Write resultsqq
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == 0:
                        #### tracking people in curent frame
                        # (x1, y1, x2, y2) = (int(xyxy[0]) * 2, int(xyxy[1]) * 2, int(xyxy[2]) * 2, int(xyxy[3]) * 2)
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
                        # cv2.imwrite(filename=save_path,img=det_img)
                        cv2.rectangle(frames, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        cv2.imwrite(filename=save_path,img=frames)
if __name__ == "__main__":
    ## load arguments
    # args = get_args()
    ### load models
    # yolov3 = YOLO(args)
    device = torch.device(f'cuda:{int(0)}') if torch.cuda.is_available() else torch.device('cpu')
    yolo_face = attempt_load('models/checkpoints/yolov5s_face.pt', map_location=device)
    extract(yolo_face, os.path.join('dataset', 'train_image_unbalanced'))
