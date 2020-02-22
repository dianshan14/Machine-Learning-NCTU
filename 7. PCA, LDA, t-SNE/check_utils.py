import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from PIL import Image
import cv2


def check_resize_resolution(path='Yale_Face_Database', subject_num=15, train=False):
    face_expression = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']

    for subject in range(1, subject_num + 1):
        for expression in face_expression:
            filename = 'subject%02d.%s.pgm' % (subject, expression)
            training_path = os.path.join(path, 'Training', filename)
            testing_path = os.path.join(path, 'Testing', filename)
            if train and os.path.isfile(training_path):
                img = plt.imread(training_path)
                py_img = cv2.pyrDown(img)
                rz_img = cv2.resize(img, dsize=py_img.shape, interpolation=cv2.INTER_CUBIC)
                print(img.shape, py_img.shape, rz_img.shape)
                show_three_imgs([img, py_img, rz_img])
            elif os.path.isfile(testing_path):
                img = plt.imread(testing_path)
                py_img = cv2.pyrDown(img)
                rz_img = cv2.resize(img, dsize=py_img.shape[::-1], interpolation=cv2.INTER_CUBIC)
                print(img.shape, py_img.shape, rz_img.shape)
                show_three_imgs([img, py_img, rz_img])


def show_three_imgs(imgs):
    plt.subplot(1,3,1)
    plt.imshow(imgs[0], cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(imgs[1], cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(imgs[2], cmap='gray')
    plt.show()
