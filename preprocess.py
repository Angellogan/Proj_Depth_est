from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os
import numpy as np
from numpy import array
import cv2
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os.path
# read images



data_path = '/home/angel/2011_09_26/DepthEstimation/'

img_path = "/home/angel/monodepth/data_in.txt"
gt = "/home/angel/monodepth/data_api.txt"

valid_f_path = '/home/angel/monodepth/top_valdata.txt'
valid_g_path = '/home/angel/monodepth/top_valgt.txt'

test_f_path = '/home/angel/monodepth/inkitti.txt'
test_g_path = '/home/angel/monodepth/ex.txt'



def normalize_x(x):

    print('shape of the input array : ', np.shape(x))

    num = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]
    depth = x.shape[3]

    mean_img = np.mean(x, axis=(0, 1, 2))
    std_dev = np.std(x) + 1e-8
    if depth is 1:
        mean_img = np.reshape(mean_img,[1,1,1])
    else:
        mean_img = np.reshape(mean_img, [1, 1, 3])

    print('mean_img',mean_img)

    norm_img = []


    for i in range(num):

        norm = ((x[i] - mean_img) / std_dev)
        norm = norm.astype(np.float32)
        print('min:', norm.min())
        print('max:', norm.max())
        norm_img.append(norm)



    return norm_img


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))





def data_process():
    imgs_list = []
    gt_list = []
    my_list = []
    val_f = []
    val_g = []
    test_f =[]
    test_g =[]

    gamma = 1.5


    with open(img_path) as f:

        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3)
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                adjusted = adjust_gamma(img1, gamma=gamma)

                norm_image = cv2.normalize(adjusted, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)



                #print(np.min(norm_image))
                #print(np.max(norm_image))
                # print(img)

                #print(np.shape(norm_image))
                norm_image1 = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)
                my_list.append(norm_image1)

    arr_img1 = np.array([np.array(xi) for xi in my_list])
    print(arr_img1.dtype)
    print("img max", np.max(arr_img1[0]))
    print("img min", np.min(arr_img1[0]))
    #arr_img1 = normalize_x(arr_img1)
    #arr_img1 = np.array([np.array(xi) for xi in arr_img1])


    #arr_img1 = np.asarray(my_list, dtype=np.float32)
    print("xtrain;", np.shape(arr_img1))
    #tf_arr1 = tf.convert_to_tensor(arr_img, dtype=tf.float32)
    #print(tf.shape(tf_arr1))

    with open(gt) as f:
        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3,cv2.IMREAD_GRAYSCALE)
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                # print(img)
                norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                #norm_image =  norm_image.astype(np.float32)
                #print(np.min(norm_image))
                #print(np.max(norm_image))
                #print(np.shape(norm_image))
                #norm_image1 = cv2.imread(norm_image, cv2.IMREAD_GRAYSCALE)

                gt_list.append(norm_image)
    arr_img2 = np.array([np.array(xi) for xi in gt_list])
    arr_img2 = np.expand_dims(arr_img2,axis=3)
    #arr_img2 = normalize_x(arr_img2)
    #arr_img2 = np.array([np.array(xi) for xi in arr_img2])

    #arr_img2 = np.asarray(gt_list, dtype=np.float32)
    print("ytrain:",np.shape(arr_img2))
    print("img max", np.max(arr_img2[0]))
    print("img min", np.min(arr_img2[0]))
    #tf_arr2 = tf.convert_to_tensor(arr_img2, dtype=tf.float32)
    #print(tf.shape(tf_arr2))


    with open(valid_f_path) as f:

        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3)
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                adjusted = adjust_gamma(img1, gamma=gamma)

                norm_image = cv2.normalize(adjusted, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #norm_image = normalize_x(adjusted)
                #norm_image = norm_image.astype(np.float32)
                #print(np.min(norm_image))
                #print(np.max(norm_image))
                # print(img)

                #print(np.shape(norm_image))
                norm_image1 = cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB)
                val_f.append(norm_image1)
    arr_img3 = np.array([np.array(xi) for xi in val_f])
    #arr_img3 = normalize_x(arr_img3)
    #arr_img3 = np.array([np.array(xi) for xi in arr_img3])
    print(arr_img3.dtype)
    #arr_img3 = np.asarray(val_f, dtype=np.float32)
    print("xval:", np.shape(arr_img3))
    #tf_arr3 = tf.convert_to_tensor(arr_img3, dtype=tf.float32)
    #print(tf.shape(tf_arr3))

    with open(valid_g_path) as f:
        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3,cv2.IMREAD_GRAYSCALE)
                #print(np.shape(img))
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                # print(img)
                norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #norm_image = normalize_x(img1)
                #norm_image = norm_image.astype(np.float32)
                #print(np.shape(norm_image))
                #print(np.min(norm_image))
                #print(np.max(norm_image))
                #print(np.shape(norm_image))
                val_g.append(norm_image)
    arr_img4 = np.array([np.array(xi) for xi in val_g])
    arr_img4 = np.expand_dims(arr_img4, axis=3)
    #arr_img4 = normalize_x(arr_img4)
    #arr_img4 = np.array([np.array(xi) for xi in arr_img4])
    print(arr_img4.dtype)
    #arr_img4 = np.asarray(val_g, dtype=np.float32)
    print("yval:",np.shape(arr_img4))
    #tf_arr4 = tf.convert_to_tensor(arr_img4, dtype=tf.float32)
    #print(tf.shape(tf_arr4))


    with open(test_f_path) as f:

        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3)
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
                adjusted = adjust_gamma(img1, gamma=gamma)

                norm_image = cv2.normalize(adjusted, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #norm_image = normalize_x(adjusted)
                #norm_image = norm_image.astype(np.float32)
                #print(np.min(norm_image))
                #print(np.max(norm_image))
                # print(img)

                #print(np.shape(norm_image))
                norm_image1 = cv2.cvtColor(norm_image,cv2.COLOR_BGR2RGB)
                test_f.append(norm_image1)
    arr_img5 = np.array([np.array(xi) for xi in test_f])
    #arr_img5 = normalize_x(arr_img5)
    #arr_img5 = np.array([np.array(xi) for xi in arr_img5])

    print(arr_img5.dtype)
    #arr_img5 = np.asarray(test_f, dtype=np.float32)
    print(np.shape(arr_img5))
    #tf_arr5 = tf.convert_to_tensor(arr_img5, dtype=tf.float32)
    #print(tf.shape(tf_arr3))

    with open(test_g_path) as f:
        for line in f:
            line2 = os.path.join(data_path, line)
            line3 = line2.rstrip()
            #print(os.path.exists(line3))
            #print(line3)
            if line:
                img = cv2.imread(line3,cv2.IMREAD_GRAYSCALE)
                #print(np.shape(img))
                img1 = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
                # print(img)
                norm_image = cv2.normalize(img1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #norm_image = normalize_x(img1)
                #norm_image = norm_image.astype(np.float32)
                #print(np.shape(norm_image))
                #print(np.min(norm_image))
                #print(np.max(norm_image))
                #print(np.shape(norm_image))
                test_g.append(norm_image)
    arr_img6 = np.array([np.array(xi) for xi in test_g])
    arr_img6 = np.expand_dims(arr_img6, axis=3)
    #arr_img6 = normalize_x(arr_img6)
    #arr_img6 = np.array([np.array(xi) for xi in arr_img6])
    #arr_img6 = np.asarray(test_g, dtype=np.float32)
    print(arr_img6.dtype)
    print(np.shape(arr_img6))
    #tf_arr6 = tf.convert_to_tensor(arr_img6, dtype=tf.float32)
    #print(tf.shape(tf_arr4))

    #tf.get_default_graph().as_graph_def()

    return arr_img1,arr_img2,arr_img3,arr_img4,arr_img5,arr_img6