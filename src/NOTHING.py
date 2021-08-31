import os
import random

import numpy as np
import torch
from cv2 import cv2
from tqdm import tqdm

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
np.random.seed(1542)

from run_yolo import image_loader, get_model

gender_choice = {0: 'female', 1: 'male'}
age_choice = {0: '10', 1: '20', 2: '30', 3: '40', 4: '50'}

def get_faces_path(path):
    paths = []
    print(f'Loading image paths.. \n')
    for dir_path,last_dir,files in tqdm(os.walk(path)):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                paths.append(os.path.join(dir_path,file))
    return paths


from utils.util import save_raw_video

if __name__ == '__main__':


    path=r'rtsp://itaz:12347890@192.168.0.58:554/stream_ch00_0'
    output_path = 'output_raw.avi'
    save_raw_video(path, output_path)
    # ### load ai hub data and put them into dataloader
    # path = r'dataset/test'
    # # path = r'G:\data\images\faces\AIhub_face\High_resolution'
    # paths_ = get_faces_path(path)
    #
    # gender_model,age_model = get_model()
    #
    # for image_path in tqdm(paths_):
    #     frames = cv2.imread(image_path)
    #     # Predict Gender and Age
    #     with torch.no_grad():
    #         #### preparing face vector before passing to age, gender estimating model
    #         vectors = image_loader(image_path).to(torch.device('cpu'))
    #         # age_gender_output = age_gender_model(vectors)
    #         pred_gender = gender_model(vectors)
    #         pred_age = age_model(vectors)
    #         ## convert predicted index to meaningful label
    #         gender_indicate = pred_gender.argmax(dim=1).item()
    #         age_indicate = round(float(pred_age))
    #         curr_gender = gender_choice.get(gender_indicate)
    #         curr_age = age_choice.get(age_indicate)
    #         print('')
    #         print('')
    #         print(f'{image_path}: gender {curr_gender} -- age {curr_age}')
    #         print('')
