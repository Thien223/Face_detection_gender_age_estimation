import argparse
from random import randint
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import time

def img_transform(image_filepaths, age):
    new_img=None
    for img in image_filepaths:
        im = Image.open(img)#file path here
        width = im.size[0]
        height = im.size[1]
        _mean = np.asarray(im).mean(axis=(0, 1, 2)) / 255
        _std = np.asarray(im).std(axis=(0, 1, 2)) / 255
        rand = randint(1,11)

        # transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std)])
        transformer = transforms.RandomChoice([transforms.RandomHorizontalFlip(p=1),
                                transforms.RandomAffine(35),
                                transforms.RandomCrop((width - int(rand ** 0.8), height- int(rand ** 0.8))),
                                transforms.RandomGrayscale(p=1),
                                transforms.RandomPerspective(),
                                transforms.RandomRotation(35, expand=False),
                                transforms.RandomRotation(35, expand=True),
                                transforms.ColorJitter(brightness=(0.2, 3)),
                                transforms.ColorJitter(contrast=(0.2, 3)),
                                transforms.ColorJitter(saturation=(0.2, 3)),
                                transforms.ColorJitter(hue=(-0.5, 0.5)),
                                transforms.ColorJitter(brightness=(0.2, 2), contrast=(0.3, 2),saturation=(0.2, 2),hue=(-0.3, 0.3)),
                                transforms.Resize((100, 100))])
        new_img = transformer(im)
        new_img_path_male = 'dataset\exported\\' + str(age) + '\\111'
        new_img_path_female = 'dataset\exported\\' + str(age) + '\\112'
        os.makedirs(new_img_path_male, exist_ok=True)
        os.makedirs(new_img_path_female, exist_ok=True)

        new_img_path_male= os.path.join(new_img_path_male, str(time.time())+'-'+img.split('\\')[-1])
        new_img_path_female= os.path.join(new_img_path_female, str(time.time())+'-'+img.split('\\')[-1])
        if '\\111\\' in img:
            print(f'MALE {img}')
            print(f'{new_img_path_male}\n\n')
            new_img = new_img.save(new_img_path_male)
        else:
            print(f'FEMALE {img}')
            print(f'{new_img_path_female}\n\n')
            new_img = new_img.save(new_img_path_female)
    return new_img

if __name__ =='__main__':
    all_face={}
    max_count = 4000
    for (dir,_,files) in os.walk(r'dataset/tarball-with-mask'):
        for img in files:
            if not img.endswith('.jpg'):
                continue
            full_path = os.path.join(dir, img)
            _,age, gender, filename = full_path.split('\\')
            age = int(age)
            gender = "male" if gender == "111" else "female"

            if age in all_face.keys():
                if gender in all_face[age].keys():
                    all_face[age][gender].append(full_path)
                else:
                    all_face[age][gender] = [full_path]
            else:
                all_face[age]={}
                all_face[age][gender] = [full_path]

    for age_ in all_face.keys():
        for gender_ in all_face[age_].keys():
            loop = int(max_count/len(all_face[age_][gender_]))

            for i in range(loop):
                print(f'age {age_} - gender {gender_} - loop {i}')
                new_imgs = img_transform(all_face[age_][gender_], age_)


