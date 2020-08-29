# coding: utf-8

from __future__ import division, print_function

import cv2
import random
import numpy as np

def get_color_table(class_num, seed=2):
    random.seed(seed)
    color_table = {}
    for i in range(class_num):
        color_table[i] = [random.randint(0, 255) for _ in range(3)]
    return color_table


def plot_one_box(img, img1, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.005 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]

    coord = deal_coord(img, coord)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img1[int(coord[1]) : int(coord[3]), int(coord[0]) : int(coord[2])]

    # image = image.astype(np.float32)

    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image

def deal_coord(img, coord):

    h, w, _ = img.shape
    h0 = h*0.05
    w0 = w*0.05
    for k, each in enumerate(coord):

        if k == 0:
            each = each - w0
            if each < 0:
                each = w0

        elif k == 1:
            each = each - h0
            if each < 0:
                each = h0


        elif k == 2 :
            each = each + w0
            if each > w:
                each = w-w0


        elif k == 3 :
            each = each + h0
            if each > h:
                each = h - h0

        coord[k] = each

    return coord