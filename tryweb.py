# from PIL import Image
import flask
from flask import request, Flask, redirect, url_for
import io
from PIL import Image
import tensorflow as tf
import TFfun
import tensorflow.contrib.slim as slim
from inception_preprocessing import preprocess_for_eval, base64_to_image, image_to_base64
from pred_single_image import inference, inference1
from test_single_image import test_one_pic, shut_sess
import numpy as np
import cv2
import time


model_name = ["Inception_V4", "ResNet50"]
clf_name = ["SVM", "KNN", "Guassian_Naive_Bayes"]
active_model = {"inception_v4": False, "resnet_v2": False}
active_clf = {"SVM" : False, "KNN" : False, "GNB" : False}

cur_page = True
image_size = 299
image = 0
image_to_pre = None
image_got = False

ckpt_dir = [r'./model/inception_v4.ckpt', r'./model/resnet_v2_50.ckpt',
            r'./model/best1', r'./model/rbest1']



def set_model_clf(fe, clf):
    global active_model, active_clf
    model_keys = list(active_model.keys())
    model_id = model_name.index(fe)

    for i in range(len(model_keys)):
        if i == model_id:
            active_model[model_keys[i]] = True
        else:
            active_model[model_keys[i]] = False

    if clf != None:
        clf_keys = list(active_clf.keys())
        clf_id = clf_name.index(clf)

        for i in range(len(clf_keys)):
            if i == clf_id:
                active_clf[clf_keys[i]] = True
            else:
                active_clf[clf_keys[i]] = False



def load_model(fe):
    global graph1
    global nets,x,sess1
    global active_clf, active_model

    if fe == "Inception_V4":
        if not active_model["inception_v4"]:
            print('---------loading---------')

            graph1 = tf.Graph()
            with graph1.as_default():

                x = tf.placeholder(tf.string)
                tf_image = preprocess_for_eval(x,image_size,image_size)
                nets = TFfun.inference_inception_v4(tf_image)

                load_fun = slim.assign_from_checkpoint_fn(
                    ckpt_dir[0],
                    TFfun.get_restore_variables([]),
                    ignore_missing_vars=True)
            sess1 = tf.Session(graph=graph1)
            return load_fun
        else:
            print('---------notloading---------')
            return None

    elif fe == "ResNet50":
        if not active_model["resnet_v2"]:

            graph1 = tf.Graph()
            with graph1.as_default():

                x = tf.placeholder(tf.string)
                tf_image = preprocess_for_eval(x, image_size, image_size)
                nets = TFfun.inference_resnet_v2(tf_image)
                load_fun = slim.assign_from_checkpoint_fn(
                    ckpt_dir[1],
                    TFfun.get_restore_variables([]),
                    ignore_missing_vars=True)

            sess1 = tf.Session(graph= graph1)
            return load_fun
        else:
            return None


def load_model1(fe):

    global active_model, graph2, image_size
    global nets1, sess2, x1

    if fe == "Inception_V4":
        if not active_model["inception_v4"]:
            graph2 = tf.Graph()

            with graph2.as_default():

                x1 = tf.placeholder(tf.string)
                image1 = preprocess_for_eval(x1, image_size, image_size)
                nets1 = inference(image1, is_training=False)

                nets1 = tf.reshape(nets1, [-1])
                saver = tf.train.Saver()

                sess2 = tf.Session(graph=graph2)

                init = tf.global_variables_initializer()
                sess2.run(init)
                saver.restore(sess2, ckpt_dir[2])

                return True
        else:
                return False

    elif fe == "ResNet50":
        if not active_model["resnet_v2"]:
            graph2 = tf.Graph()

            with graph2.as_default():

                x1 = tf.placeholder(tf.string)
                image1 = preprocess_for_eval(x1, image_size, image_size)
                nets1 = inference1(image1, is_training=False)

                nets1 = tf.reshape(nets1, [-1])
                saver = tf.train.Saver()

                sess2 = tf.Session(graph=graph2)

                init = tf.global_variables_initializer()
                sess2.run(init)
                saver.restore(sess2, ckpt_dir[3])

                return True
        else:
            return False



app = Flask(__name__)
app.debug = True

@app.route("/")
def page():
    global sess1, sess2
    sess1 = sess2 = None
    print(request.remote_addr)
    return flask.render_template("fpage.html")


@app.route("/return1", methods=["GET"])
def ret1():
    for each in active_model.keys():
        active_model[each] = False

    if sess1 != None:
        sess1.close()
    return redirect(url_for('page'))


@app.route("/return2", methods=["GET"])
def ret2():
    print('haha')
    for each in active_model.keys():
        active_model[each] = False
    if sess2 != None:
        sess2.close()

    return redirect(url_for('page'))


@app.route("/jump", methods=["POST"])
def jump():
    global cur_page
    if request.method == "POST":
        page = request.values.get("ra")
        page = str(page).strip()
        if page == "r1":
            cur_page = True
            return  flask.render_template('myweb.html', pred = " ")
        else:
            cur_page = False
            return flask.render_template('myweb1.html', pred=" ")
    else:
        return "error!"


@app.route("/upload_image", methods=["POST"])
def pprint():
    global image, image_to_pre
    global image_got
    global obj_ornot
    image_to_pre = None
    if request.method == "POST":
        obj_ornot = request.values.get("obj")
        obj_ornot = str(obj_ornot).strip()

        if request.files.get("image"):
            image_got = True
            image = request.files["image"].read()
            if obj_ornot == "obj":
                obj_ornot = True
                img_obj = io.BytesIO(image)
                img_obj = Image.open(img_obj)
                img_obj = img_obj.convert("RGB")
                img_obj.save(r'./img_obj.jpg')
                # image_obj = base64_to_image(image)
                image_to_pre, image_obj = test_one_pic()
                image_obj = image_to_base64(image_obj)
                return flask.render_template('myweb1.html', pred=" ", img_stream=image_obj)

            else:
                img_obj = io.BytesIO(image)
                img_obj = Image.open(img_obj)
                img_obj = img_obj.convert("RGB")
                img_obj.save(r'./img_obj.jpg')
                obj_ornot = False

        else:
            image_got = False
            print('None')

    if cur_page:
        return flask.render_template('myweb.html', pred = " ")
    else:
        return flask.render_template('myweb1.html', pred=" ")



@app.route("/pre1", methods=["POST"])
def pre1():
    global image, active_model, active_clf
    global image_got, graph1, x, nets, sess1
    if image_got:
        if request.method == "POST":
            fe = request.form.get("features_extractor")
            clf = request.form.get("classifier")
            fe = str(fe.strip())
            clf = str(clf.strip())

            load_fun = load_model(fe)
            set_model_clf(fe, clf)

            with graph1.as_default():
                if load_fun != None:
                    init = tf.global_variables_initializer()
                    sess1.run(init)
                    load_fun(sess1)

                # image = TFfun.preprocessing(image, image_size[model_name.index(fe)])
            logits = sess1.run(nets, feed_dict = {x: image})

            pred = TFfun.skclf_predict(active_model, active_clf, logits)

            image_got = False
            pred = fe +' and '+ clf + ' get --->  '+ pred
            return flask.render_template('myweb.html', pred = pred)

        else:
            print("Get None")
            return flask.render_template('myweb.html', pred=" ")
    else:
        ###don't know if it works (render_template or redirect)
        return flask.render_template('myweb.html', pred = " ")



@app.route("/pre2", methods = ["POST"])
def pre2():
    global image_got, image_to_pre
    global graph2, sess2, obj_ornot, nets1
    if image_got:
        if request.method == "POST":
            fe = request.form.get("features_extractor")
            fe = str(fe.strip())

            load_model1(fe)
            set_model_clf(fe, None)

            if obj_ornot:

                while image_to_pre == None:
                    time.sleep(0.1)

                breed = []
                for k, each in enumerate(image_to_pre):

                    # cv2.imshow('1', each)
                    # cv2.waitKey(0)
                    cv2.imwrite(r'./img_obj.jpg', each)
                    # time.sleep(0.2)
                    pic = tf.gfile.FastGFile(r'./img_obj.jpg', 'rb').read()
                    pred = sess2.run(nets1, feed_dict={x1 : pic})
                    pred = np.argsort(pred)

                    breed.append((k, TFfun.id_map_breed[pred[-1]]))

                pred = fe + ' get --->  ' + str(breed)
                obj_ornot = False
                return flask.render_template('myweb1.html', pred=pred)

            else:
                pic = tf.gfile.FastGFile(r'./img_obj.jpg', 'rb').read()
                pred = sess2.run(nets1, feed_dict={x1 : pic})
                pred = np.argsort(pred)

                pred = TFfun.id_map_breed[pred[-1]]
                pred = fe  + ' get --->  '+ str(pred)
                return flask.render_template('myweb1.html', pred = pred)
        else:
            print("Get None")
            return flask.render_template('myweb1.html', pred=" ")

        image_got = False
    else:
        return flask.render_template('myweb1.html', pred = " ")


if __name__ == "__main__":

    app.run(host='192.168.88.101', port=8000)
    shut_sess()
    # app.run()


