import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os 
import cv2
import numpy as np
from scipy.misc import imresize, imsave, imread


def crop_image(path, save_dir):
    # path = 'D:\\UNIT-master\\datasets\\day2night\\testB'

    img_list = os.listdir(path)
    for img_name in img_list:
        img_address = os.path.join(path, img_name)
        img = imread(img_address)
        img = img[80:, :]
        img = imresize(img, (512, 1024))
        img = img[0:426, 192:760]
        img = imresize(img, (224, 224))
        imsave(os.path.join(save_dir, img_name), img)

def change_name():
    path = 'follow_up_test_set/add_car'
    image_list = os.listdir(path)

    for i, file_name in enumerate(image_list):
        dst = str(i) + '.jpg'
        src = os.path.join(path, file_name)
        dst = os.path.join(path, dst)

        os.rename(src, dst)