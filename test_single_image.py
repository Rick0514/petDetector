# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from pred_single_image import test_pic

from model import yolov3

graph2 = None

def test_one_pic():
    global graph2, sess3, boxes, scores, labels, input_data

    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    parser.add_argument("--input_image", type=str,default='./img_obj.jpg',
                        help="The path of the input image.")
    parser.add_argument("--anchor_path", type=str, default="./data/my_data/anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/my_data/data.name",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/best_model_Epoch_38_step_16808_mAP_0.9421_loss_1.3932_lr_3e-05",
                        help="The path of the weights to restore.")
    args = parser.parse_args()

    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)

    color_table = get_color_table(args.num_class)

    img_ori = cv2.imread(args.input_image)
    img1 = img_ori.copy()
    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    if graph2 == None:
        graph2 = tf.Graph()
        with graph2.as_default():
            input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
            yolo_model = yolov3(args.num_class, args.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

            saver = tf.train.Saver()
            sess3 = tf.Session(graph=graph2)
            saver.restore(sess3, args.restore_path)


    boxes_, scores_, labels_ = sess3.run([boxes, scores, labels], feed_dict={input_data: img})


    # rescale the coordinates to the original image
    if args.letterbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

    image = []

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        image0 = plot_one_box(img_ori, img1, [x0, y0, x1, y1],
                             label=str(i) + ' ' + args.classes[labels_[i]] +
                             ', {:.2f}%'.format(scores_[i] * 100),
                             color=color_table[labels_[i]])

        image.append(image0)

    img_ori = cv2.resize(img_ori, (300,300))
    return image, img_ori


def shut_sess():
    sess3.close()
# breed = test_pic(image, 'res')
# print(breed)
#
# cv2.imshow('Detection result', img_ori)
# cv2.imwrite('detection_result.jpg', img_ori)
# cv2.waitKey(0)