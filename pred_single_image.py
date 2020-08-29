import sys
nets_dir=r'E:\Anaconda_new\Lib\site-packages\tensorflow\contrib\slim\python\slim'
sys.path.append(nets_dir)
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v4 as inception_v4
from inception_preprocessing import preprocess_for_eval
import tensorflow.contrib.slim.python.slim.nets.resnet_v2 as resnet_v2
import pickle
import cv2


n_class = 37
id_file = r'E:\learn_vision\dogs_and_cats\id_map_breed.pickle'
model_inv4 = r'C:\Users\Rick\Desktop\cnn\tuned_model\inv4\best1'
model_res = r'C:\Users\Rick\Desktop\cnn\tuned_model\res\rbest1'


def get_tuned_variables():
    # variable_to_restore = []
    # exclusions = []

    # for var in slim.get_model_variables():
    #     judge = True
    #     for ex in exclusions:
    #         if var.op.name.startswith(ex):
    #             judge = False
    #             break
    #     if judge:
    #         variable_to_restore.append(var)
    variable_to_restore = slim.get_model_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

    variable_to_restore.extend(bn_moving_vars)

    return variable_to_restore


def inference(inputs,is_training):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits,_=inception_v4.inception_v4(inputs,
                                           num_classes=n_class,
                                           is_training=is_training)
    return logits

def inference1(inputs,is_training):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits,_=resnet_v2.resnet_v2_50(inputs,
                                        num_classes=n_class,
                                        is_training=is_training)

    return logits


def test_pic(img, kind = 'inv4'):
    breed = []
    with open(id_file, 'rb') as file:
        id_map_breed = pickle.load(file)

    tf.reset_default_graph()
    image_size = 299

    x = tf.placeholder(tf.uint8, [image_size, image_size, 3])
    image = preprocess_for_eval(x, image_size, image_size, central_crop = False)
    image = tf.expand_dims(image, 0)

    if kind == 'inv4':
        logits = inference(image, is_training = False)
    elif kind == 'res':
        logits = inference1(image, is_training = False)

    logits = tf.reshape(logits, [-1])

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # init = tf.global_variables_initializer()
        # sess.run(init)

        if kind == 'inv4':
            saver.restore(sess, model_inv4)
        elif kind == 'res':
            saver.restore(sess, model_res)

        for k, each in enumerate(img):
            each = cv2.resize(each, (image_size, image_size))

            feed_dict = {x : each}
            pred = sess.run(logits, feed_dict)
            pred = np.argsort(pred)
            breed.append((k, id_map_breed[pred[-1]]))

    return breed


