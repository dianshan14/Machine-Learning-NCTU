import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as distance
from PIL import Image
import cv2

def set_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3) # 2, 3, 4
    parser.add_argument('--image', type=str, default='2') # 1, 2
    parser.add_argument('--init_method', type=str, default='color_cloest_picked') # ...
    parser.add_argument('--gamma_s', type=float, default=3) # 10, 0.1, 0.001
    parser.add_argument('--gamma_c', type=float, default=3)
    parser.add_argument('--gif_loop', type=int, default=0)
    parser.add_argument('--cut_type', type=str, default='ratio_cut') # *2
    # parser.add_argument('--cut_type', type=str, default='normalized_cut')
    args = parser.parse_args()

    return args


def get_data(path='.', filename='image1.png', scaledown=False):
    img1 = plt.imread(os.path.join(path, filename))
    if scaledown:
        img1 = cv2.pyrDown(img1)
    img_size = img1.shape[0]
    color = img1.reshape(-1, 3)
    spatial = np.array([[i, j] for i in range(img_size) for j in range(img_size)])
    return img_size, img1.reshape(-1, 3), spatial


def get_kernel(spatial, color, gamma_s=0.025, gamma_c=0.025):
    spatial_RBF = np.exp(-gamma_s * distance.cdist(spatial, spatial, 'sqeuclidean'))
    color_RBF = np.exp(-gamma_c * distance.cdist(color, color, 'sqeuclidean'))
    return spatial_RBF * color_RBF


def plot(result, k, args, plot_setting=None):
    color_mapping = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                              [1, 1, 0], [0, 1, 1], [1, 0, 1],
                              [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])

    SCALE_UP_TIMES = 3
    PADDING_WIDTH = 8
    PADDING_HEIGHT = 25
    PADDING_COLOR = [47, 51, 54]
    PADDING_COLOR = [color / 255 for color in PADDING_COLOR]
    RGB_result = np.empty((result.__len__(),
                           2**SCALE_UP_TIMES*(args.img_size + PADDING_HEIGHT + PADDING_WIDTH),
                           2**SCALE_UP_TIMES*(args.img_size + 2 * PADDING_WIDTH), 3))

    HEIGHT_OFFSET = 35
    TEXT_STYLE = cv2.FONT_HERSHEY_TRIPLEX
    TEXT_WIDTH = 1
    TEXT_SCALE = 0.7
    TEXT_COLOR = [206, 217, 224]
    TEXT_COLOR = [color / 255 for color in TEXT_COLOR]
    ITER_COLOR = [34, 224, 81]
    ITER_COLOR = [color / 255 for color in ITER_COLOR]


    METHOD = plot_setting.get('method')
    TEXT = [
        METHOD,
        'INIT_METHOD: %s' % args.init_method,
        'k=%s, gamma_s=%.3f, gamma_c=%.3f' % (args.k, args.gamma_s, args.gamma_c),
        'Iteration: '
    ]
    for i, v in enumerate(result):
        low_res = cv2.copyMakeBorder(color_mapping[result[i]],
                                     PADDING_HEIGHT, PADDING_WIDTH, PADDING_WIDTH, PADDING_WIDTH,
                                     cv2.BORDER_CONSTANT, value=PADDING_COLOR)
        for j in range(SCALE_UP_TIMES):
            low_res = cv2.pyrUp(low_res)


        for m, text in enumerate(TEXT):
            cv2.putText(low_res, text, (HEIGHT_OFFSET, 60 + m * HEIGHT_OFFSET),
                        TEXT_STYLE,
                        TEXT_SCALE, TEXT_COLOR, TEXT_WIDTH, cv2.LINE_AA)

        cv2.putText(low_res, 'Iteration  ', (HEIGHT_OFFSET, 60 + m * HEIGHT_OFFSET),
                    TEXT_STYLE,
                    TEXT_SCALE, TEXT_COLOR, TEXT_WIDTH, cv2.LINE_AA)
        cv2.putText(low_res, '%s' % (i+1, ), (8 * HEIGHT_OFFSET, 60 + m * HEIGHT_OFFSET),
                    TEXT_STYLE,
                    TEXT_SCALE, ITER_COLOR, TEXT_WIDTH, cv2.LINE_AA)
        RGB_result[i] = low_res
        # RGB_result[i] = color_mapping[result[i]]
        if len(RGB_result) < 100:
            plt.imsave('%s/%s.png' % (args.PATH, i), RGB_result[i])

    # padding
    make_gif(RGB_result, args.gif_loop, args)


def make_gif(result, gif_loop, args):
    imgs = []
    for img in result:
        imgs.append(Image.fromarray(np.uint8(img*255)))
    imgs[0].save('%s/output.gif' % args.PATH, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=300, loop=gif_loop)
