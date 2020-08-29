import sys
nets_dir = r'E:\Anaconda_new\Lib\site-packages\tensorflow\contrib\slim\python\slim'
sys.path.append(nets_dir)


import tensorflow as tf
import os
from PIL import Image
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v4, resnet_v2
import pickle
from sklearn.pipeline import Pipeline

# from inception_preprocessing import preprocess_image

# 1 model_dir
model_dir = {'pca_inv4': 'pca_inv4.pkl', 'svm_inv4': 'svm_inv4.pkl',
             'pca_res': 'pca_res.pkl', 'svm_res': 'svm_res.pkl'}

for each in model_dir.keys():
    model_dir[each] = os.path.join(r'./model', model_dir[each])

# 2 breed name

with open('./model/id_map_breed.pickle', 'rb') as f:
    id_map_breed = pickle.load(f)


def preprocessing(image, img_size):

    image = image.convert("RGB")
    image = image.resize((img_size, img_size),Image.BILINEAR)

    image = np.array(image)
    image = np.reshape(image,(-1, img_size, img_size, 3))

    return image


# get the variable to be restored
def get_restore_variables(exclusions):
    variable_to_restore = []

    for var in slim.get_model_variables():
        judge = True
        for ex in exclusions:
            if var.op.name.startswith(ex):
                judge = False
                break
        if judge:
            variable_to_restore.append(var)

    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]

    variable_to_restore.extend(bn_moving_vars)

    return variable_to_restore


def inference_inception_v4(inputs):
    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits, _ = inception_v4.inception_v4(inputs, is_training=False)

    return logits

def inference_resnet_v2(inputs):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_50(inputs, num_classes = 1001, is_training = False)

    return logits


def skclf_predict(act_fe, act_clf, logits):

    if act_fe["inception_v4"]:

        with open(model_dir["pca_inv4"], 'rb') as f:
            pca = pickle.load(f)
        if act_clf["SVM"]:
            # load_model
            with open(model_dir["svm_inv4"], 'rb') as f:
                SVM = pickle.load(f)

            estimators = [('reduce_dim', pca), ('clf', SVM)]
            pipe = Pipeline(estimators)
            pred = pipe.predict(logits)
            pred = id_map_breed[int(pred)]

            return pred

    elif act_fe["resnet_v2"]:

        with open(model_dir["pca_res"], 'rb') as f:
            pca = pickle.load(f)
        if act_clf["SVM"]:
            # load_model
            with open(model_dir["svm_res"], 'rb') as f:
                SVM = pickle.load(f)

            estimators = [('reduce_dim', pca), ('clf', SVM)]
            pipe = Pipeline(estimators)
            logits = logits.reshape([1,logits.shape[3]])
            pred = pipe.predict(logits)
            pred = id_map_breed[int(pred)]

            return pred

