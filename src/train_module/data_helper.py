# -*- coding:utf-8 -*-
import os
import json
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch, numpy as np
from cv2 import cv2

torch.manual_seed(1542)
torch.cuda.manual_seed(1542)
torch.backends.deterministic = True
torch.backends.benchmark = False
random.seed(1542)
img_size = 64

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

#
# def image_preprocess(img_path, train=True):
    # from skimage.feature import local_binary_pattern
    # from utils.util import lbp
    # ### load image using opencv
    # if os.path.isfile(img_path):
    #     img = cv2.imread(img_path)
    # else:
    #     img = img_path
    # ### convert to Lab color format
    # lab_img = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
    #
    # R=3
    # P=R*8
    # lbp_img=np.zeros_like(img)
    # for i in range(img.shape[-1]):
    #     lbp_img[:,:,i] = local_binary_pattern(img[:,:,i],P,R,method='var')
    #
    # try:
    #     lbp_img = Image.fromarray(lbp_img)
    #     lab_img = Image.fromarray(lab_img)
    # except AttributeError:
    #     raise AttributeError('Image_preprocess function only accepts image path or numpy image array..')
    # train_preprocess = transforms.Compose([transforms.Resize((img_size+6,img_size+6)),
    #                                        transforms.CenterCrop(img_size),
    #                                        transforms.AutoAugment(),
    #                                        transforms.RandomAutocontrast(),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize(mean=mean, std=std)
    #                                        ])
    #
    # val_preprocess = transforms.Compose([transforms.Resize((img_size+6,img_size+6)),
    #                                      transforms.CenterCrop(img_size),
    #                                      transforms.ToTensor(),
    #                                      transforms.Normalize(mean=mean, std=std)
    #                                      ])
    # preprocess = train_preprocess if train else val_preprocess
    # lbp_img = preprocess(lbp_img)
    # lab_img = preprocess(lab_img)
    # # img=torch.cat((lbp_img,lab_img),dim=0)
    # return lbp_img

#
#
# #
def image_preprocess(img_path, train=True):
    ### load image using opencv
    if os.path.isfile(img_path):
        img = cv2.imread(img_path)
    else:
        img = img_path
    try:
        img = Image.fromarray(img)
    except AttributeError:
        raise AttributeError('Image_preprocess function only accepts image path or numpy image array..')
    train_preprocess = transforms.Compose([transforms.Resize((img_size + 4, img_size + 4)),
                                           transforms.CenterCrop(img_size),
                                           transforms.AutoAugment(),
                                           transforms.RandomAutocontrast(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=mean, std=std)
                                           ])

    val_preprocess = transforms.Compose([transforms.Resize((img_size + 4, img_size + 4)),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)
                                         ])
    preprocess = train_preprocess if train else val_preprocess
    img = preprocess(img)
    return img
# #


def _age_categorization(age):
    if age < 20:
        age_ = 0
    elif 20 <= age < 30:
        age_ = 1
    elif 30 <= age < 40:
        age_ = 2
    elif 40 <= age < 50:
        age_ = 3
    elif 50 <= age:
        age_ = 4
    # elif 60 <= age < 70:
    #     age_=5
    # elif 70 <= age:
    #     age_=6
    else:
        age_ = -1
    return int(age_)


class FaceImageDataset(Dataset):
    def __init__(self, info):
        self.info = info
    def __getitem__(self, idx):
        temp = self.info[idx]
        img_path = temp['image']
        assert img_path is not None, f'image path can not be None: {temp}'
        img = image_preprocess(img_path)
        ### argmax(gender)==1 --> male. other wise female
        gender = torch.tensor(([0, 1]), dtype=torch.float32) if temp['gender'] == 'male' else torch.tensor(([1, 0]),dtype=torch.float32)
        age = torch.tensor(_age_categorization(temp['age']), dtype=torch.float32)
        return img, gender, age

    def __len__(self):
        return len(self.info)


def split_dataset(path: list, split_rate: list, age_balance_count=None):
    ### load tarball dataset
    # # path = ['src/dataset/train_image_unbalanced/metadata.json']
    # print(f'{path}')
    # if split_rate is None:
    #     split_rate = [0.8, 0.2]
    # unbalanced_dataset = None
    # for i in range(len(path)):
    #     with open(path[i], 'r') as img_files:
    #         data = json.load(img_files)
    #     unbalanced_dataset = data if unbalanced_dataset is None else unbalanced_dataset + data
    # random.shuffle(unbalanced_dataset)


    ### load tarball dataset
    path_tarball = 'dataset/train_image_unbalanced/metadata.json'
    path_aihub = 'dataset/aihub_unbalanced/metadata.json'
    print(f'data paths: {[path_tarball,path_aihub]}')
    if split_rate is None:
        split_rate = [0.8, 0.2]
    with open(path_tarball, 'r') as img_files:
        data_tarball = json.load(img_files)

        ### only get 10대 data from tarball (aihub data does not have 10대 data)
    data_tarball = list(filter(lambda x: x['age'] == 10, data_tarball))

    with open(path_aihub, 'r') as img_files:
        data_aihub = json.load(img_files)
    unbalanced_dataset = data_tarball + data_aihub
    random.shuffle(unbalanced_dataset)

    #### set age_balance_count to None to use all data (not limit to balanced age)
    male_10_set = list(filter(lambda x: x['gender'] == 'male' and x['age'] == 10, unbalanced_dataset))[
                  :age_balance_count]
    male_20_set = list(filter(lambda x: x['gender'] == 'male' and x['age'] == 20, unbalanced_dataset))[
                  :age_balance_count]
    male_30_set = list(filter(lambda x: x['gender'] == 'male' and x['age'] == 30, unbalanced_dataset))[
                  :age_balance_count]
    male_40_set = list(filter(lambda x: x['gender'] == 'male' and x['age'] == 40, unbalanced_dataset))[
                  :age_balance_count]
    male_50_set = list(filter(lambda x: x['gender'] == 'male' and x['age'] == 50, unbalanced_dataset))[
                  :age_balance_count]

    female_10_set = list(filter(lambda x: x['gender'] == 'female' and x['age'] == 10, unbalanced_dataset))[
                  :age_balance_count]
    female_20_set = list(filter(lambda x: x['gender'] == 'female' and x['age'] == 20, unbalanced_dataset))[
                  :age_balance_count]
    female_30_set = list(filter(lambda x: x['gender'] == 'female' and x['age'] == 30, unbalanced_dataset))[
                  :age_balance_count]
    female_40_set = list(filter(lambda x: x['gender'] == 'female' and x['age'] == 40, unbalanced_dataset))[
                  :age_balance_count]
    female_50_set = list(filter(lambda x: x['gender'] == 'female' and x['age'] == 50, unbalanced_dataset))[
                  :age_balance_count]
    print('================ Female ==================')
    print(len(female_20_set))
    print(len(female_30_set))
    print(len(female_40_set))
    print(len(female_50_set))
    print(len(female_10_set))
    print('============================================')
    print('================ Male ==================')
    print(len(male_20_set))
    print(len(male_30_set))
    print(len(male_40_set))
    print(len(male_50_set))
    print(len(male_10_set))
    print('============================================')

    male_set =  male_10_set+male_20_set + male_30_set + male_40_set + male_50_set
    female_set = female_10_set+female_20_set + female_30_set + female_40_set + female_50_set

    gender_max_count = min(len(male_set), len(female_set))
    male_set = male_set[:gender_max_count]
    female_set = female_set[:gender_max_count]
    dataset = male_set + female_set
    random.shuffle(dataset)
    size = len(dataset)
    train_set = dataset[0: int(size * split_rate[0])]
    validation_set = dataset[int(size * (split_rate[0])):]
    print(f"malecount: {len([a['gender'] == 'male' for a in train_set])}")
    print(f"femalecount: {len([a['gender'] == 'female' for a in train_set])}")
    return train_set, validation_set

# #
# #
# #
# def save_tensor_as_image(tensor, path):
#     from cv2 import cv2
#     ### tensor should be (c,h,w) format
#     ##f"val_images/{time.time()}_{int(torch.argmax(gender, dim=1))}_{int(age)}.png"
#     assert path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'), 'pass correct image file format to the save path'
#     tensor = denormalizing(tensor, mean=[0, 0, 0], std=[255, 255, 255])
#     img_ = cv2.cvtColor(tensor.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
#     cv2.imwrite(path, img_)

# from torch.utils.data import DataLoader
#
# img_path = [r'dataset/train_image_unbalanced/metadata.json', r'dataset/train_image_unbalanced/metadata.json']
#
# train, val = split_dataset(img_path, split_rate=[0.1, 0.1])
# from cv2 import cv2
# import time, numpy as np
#
# dataset = FaceImageDataset(train)
# val_data_loader = DataLoader(dataset=dataset, batch_size=1)
# count = 0
# for img, gender, age in val_data_loader:
# 	# print(img)
# 	img = img.squeeze(0)
# 	img = denormalizing(img,mean=[0,0,0],std=[255,255,255])
# 	#
# 	# print(img)
# 	# break
# 	count += 1
# 	if count > 5: break
# 	img_ = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
# 	cv2.imwrite(f"val_images/{time.time()}_{int(torch.argmax(gender, dim=1))}_{int(age)}.png", img_)
# cv2.imshow('img',img_)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# img_path=r'G:\locs_projects\on_working\Face_detection\val_images\10579A36-with-mask.jpg'
# img = Image.open(img_path)
# preprocess = transforms.Compose([transforms.Resize(img_size + 4),
#                                  transforms.RandomCrop(img_size),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  # transforms.Normalize(mean=mean,
#                                  #                      std=std)
#                                  ])
# print(np.array(img))
# # img = np.array(img)
# img_ = preprocess(img)
# torch.min(img_)
# torch.max(img_)
# print(img_)
# img = denormalizing_(img_,mean=[0,0,0],std=[255,255,255])
# torch.min(img)
# torch.max(img)

#
#

# #
# def denormalizing__(tensor):
#     inv_normalize = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0,0,0],std=[255,255,255])
#                                         ])
#     return inv_normalize(tensor)
