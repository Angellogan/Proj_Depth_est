from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import depth_utils
import os

PLOT_DIR = '/home/angel/monodepth/OP_plots/plots'

is_training = True
sess = tf.get_default_session()

def normalize(x):
    print('dtpe', np.dtype(x))
    norm_op = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_op


def batch_normalization(input_data, is_training, scale_offset=True, elu=True):
    #print(sess.run(tf.equal(is_training, tf.constant(True))))
    name = tf.get_variable_scope().name


    with tf.variable_scope(name) as scope:
        shape = [input_data.get_shape()[-1]]
        pop_mean = tf.get_variable("mean", shape, initializer=tf.constant_initializer(0.0), trainable=False)
        pop_var = tf.get_variable("variance", shape, initializer=tf.constant_initializer(1.0), trainable=False)
        epsilon = 1e-4
        decay = 0.999
        if scale_offset:
            print('inside scale offset')
            scale = tf.get_variable("scale", shape, initializer=tf.constant_initializer(1.0))
            offset = tf.get_variable("offset", shape, initializer=tf.constant_initializer(0.0))
        else:
            print('outside scale offset')
            scale, offset = (None, None)



        def train_batch_norm():
            print("Training bAtch Norm")
            batch_mean, batch_var = tf.nn.moments(input_data, [0, 1, 2])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                output = tf.nn.batch_normalization(input_data,
                                                   batch_mean, batch_var, offset, scale, epsilon, name=name)

                return output
        '''   
        if tf.equal(is_training,tf.constant(True)):

            print('training batch norm')
           

            batch_mean, batch_var = tf.nn.moments(input_data, [0, 1, 2])

            train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                output = tf.nn.batch_normalization(input_data,
                                               batch_mean, batch_var, offset, scale, epsilon, name=name)
                #output = tf.nn.elu(output)
        else:
            print('test batch norm')
            output = tf.nn.batch_normalization(input_data,
                                               pop_mean, pop_var, offset, scale, epsilon, name=name)
        '''


        def test_batch_norm():
            print("Testing  batch norm")
            output = tf.nn.batch_normalization(input_data,
                                               pop_mean, pop_var, offset, scale, epsilon, name=name)
            return output


        output = tf.cond(tf.equal(is_training, tf.constant(True)), train_batch_norm, test_batch_norm)




        output = tf.nn.elu(output)
        return  output






def conv(x, in_layers, num_out_layers, kernel_size, stride):
    print('inside conv')

    kernel = [kernel_size, kernel_size, in_layers, num_out_layers]
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    # x = tf.Print(x, [tf.shape(x)], message="shape of the image before padding")
    # p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    # print("shape od p_x;", tf.shape(p_x))
    # p_x = tf.Print(p_x, [tf.shape(p_x), p_x], message="After Padding")

    weights = tf.get_variable("weights", kernel, initializer=tf.truncated_normal_initializer(0.001))
    tf.add_to_collection('conv_weights', weights)
    # filter_summary = tf.image_summary(filter)
    conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding="SAME")
    bias = tf.Variable(tf.constant(0.01, shape=[num_out_layers], dtype=tf.float32),
                       trainable=True, name='biases')
    z = tf.nn.bias_add(conv, bias)
    #print(sess.run(is_training))
    activations = batch_normalization(z,is_training)
    #activations = tf.nn.relu(z)
    tf.add_to_collection('conv_output', activations)

    # output = slim.conv2d(p_x, num_out_layers, kernel, stride, 'VALID', activation_fn=activation_fn)

    return activations


def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    #x = np.asarray(x)
    #x = x.astype(np.float32, copy = False)
    return tf.image.resize_bilinear(x, [h * ratio, w * ratio])
    #return cv2.resize(x,[h * ratio, w * ratio], interpolation= cv2.INTER_LINEAR)


def upconv(x, in_layers, num_out_layers, kernel_size, scale):
    print('inside upconv')
    upsample = upsample_nn(x, scale)
    conv_op = conv(upsample, in_layers, num_out_layers, kernel_size, 1)
    return conv_op


def accuracy(fconv3,ground_truth):
    pass

def vgg(image, ground_truth):

    #print(sess.run(tf.equal(is_training,tf.constant(True))))
    print("model is called")
    image =  tf.Print(image, [tf.shape(image)], message= 'shape of input image inside model')
    tf.add_to_collection('input_images', image)

    with tf.variable_scope("encoder") as scope:

        with tf.variable_scope('conv1'):
            image = tf.Print(image,[tf.reduce_max(image), tf.reduce_min(image)], message = "max and min image")

            print(type(image))
            conv1 = conv(image, 3, 32, 7, 2)
            conv1 = tf.Print(conv1,[tf.reduce_max(conv1), tf.reduce_min(conv1)], message = "max and min conv1")

            # conv1 = tf.Print(conv1, [tf.shape(conv1)], message="shape pf conv1")
        with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
            conv2 = conv(conv1, 32, 64, 5, 2)
            conv2 = tf.Print(conv2, [tf.reduce_max(conv2), tf.reduce_min(conv2)], message="max and min conv2")
            #conv2 = tf.Print(conv2, [tf.shape(conv2)], message="shape pf conv1")
        with tf.variable_scope('conv3'):
            conv3 = conv(conv2, 64, 64, 3, 2)
            conv3 = tf.Print(conv3, [tf.reduce_max(conv3), tf.reduce_min(conv3)], message="max and min conv3")
            #conv3 = tf.Print(conv3, [tf.shape(conv3)], message="shape pf conv3")
        with tf.variable_scope('conv4'):
            conv4 = conv(conv3, 64, 64, 3, 2)
            conv4 = tf.Print(conv4, [tf.reduce_max(conv4), tf.reduce_min(conv4)], message="max and min conv4")
            #conv4 = tf.Print(conv4, [tf.shape(conv4)], message="shape pf conv4")
        with tf.variable_scope('conv5'):
            conv5 = conv(conv4, 64, 128, 3, 2)
            conv5 = tf.Print(conv5, [tf.reduce_max(conv5), tf.reduce_min(conv5)], message="max and min conv5")
            #conv5 = tf.Print(conv5, [tf.shape(conv5)], message="shape pf conv5")
        with tf.variable_scope('conv6'):
            conv6 = conv(conv5, 128, 128, 3, 2)
            conv6 = tf.Print(conv6, [tf.reduce_max(conv6), tf.reduce_min(conv6)], message="max and min conv6")
            #conv6 = tf.Print(conv6, [tf.shape(conv6)], message="shape pf conv6")
        with tf.variable_scope('conv7'):
            conv7 = conv(conv6, 128, 128, 3, 2)
            conv7 = tf.Print(conv7, [tf.reduce_max(conv7), tf.reduce_min(conv7)], message="max and min conv7")
            #conv7 = tf.Print(conv7, [tf.shape(conv7)], message="shape pf conv7")

    with tf.variable_scope("decoder"):
        print("inside decoder")
        with tf.variable_scope("iconv7"):
            iconv7 = upconv(conv7, 128, 128, 3, 2)
            iconv7 = tf.Print(iconv7, [tf.reduce_max(iconv7), tf.reduce_min(iconv7)], message="max and min iconv7")
        with tf.variable_scope("iconv6"):
            iconv6 = upconv(iconv7, 128, 128, 3, 2)
            iconv6 = tf.Print(iconv6, [tf.reduce_max(iconv6), tf.reduce_min(iconv6)], message="max and min iconv6")
        with tf.variable_scope("iconv5"):
            iconv5 = upconv(iconv6, 128, 128, 3, 2)
            iconv5 = tf.Print(iconv5, [tf.reduce_max(iconv5), tf.reduce_min(iconv5)], message="max and min iconv5")
        with tf.variable_scope("iconv4"):
            iconv4 = upconv(iconv5, 128, 64, 3, 2)
            iconv4 = tf.Print(iconv4, [tf.reduce_max(iconv4), tf.reduce_min(iconv4)], message="max and min iconv4")
        with tf.variable_scope("iconv3"):
            iconv3 = upconv(iconv4, 64, 64, 3, 2)
            iconv3 = tf.Print(iconv3, [tf.reduce_max(iconv3), tf.reduce_min(iconv3)], message="max and min iconv3")
        with tf.variable_scope("iconv2",is_training):
            iconv2 = upconv(iconv3, 64, 64, 3, 2)
            iconv2 = tf.Print(iconv2, [tf.reduce_max(iconv2), tf.reduce_min(iconv2)], message="max and min iconv2")
        with tf.variable_scope("iconv1"):
            iconv1 = upconv(iconv2, 64, 32, 3, 2)
            iconv1 = tf.Print(iconv1, [tf.reduce_max(iconv1), tf.reduce_min(iconv1)], message="max and min iconv1")

    with tf.variable_scope("fine-tune"):
        with tf.variable_scope("Ft-1"):
            fconv1 = conv(image, 3, 32, 3, 1) #867
            fconv1 = tf.Print(fconv1, [tf.reduce_max(fconv1), tf.reduce_min(fconv1)], message="max and min fconv1")

            fconv1_cat = tf.concat([fconv1, iconv1], 3)
            fconv1_cat = tf.Print(fconv1_cat, [tf.reduce_max(fconv1_cat), tf.reduce_min(fconv1_cat)], message="max and min fconv1_cat")

        with tf.variable_scope("Ft-2"):
            fconv2 = conv(fconv1_cat, 64, 16, 5, 1) #25616
            fconv2 = tf.Print(fconv2, [tf.reduce_max(fconv2), tf.reduce_min(fconv2)], message="max and min fconv2")

        with tf.variable_scope("Ft-3"):
            fconv3 = conv(fconv2, 16, 1, 5, 1) #85 learnable param
            fconv3 = tf.Print(fconv3, [tf.reduce_max(fconv3), tf.reduce_min(fconv3)], message="max and min fconv3")


    # Add fc here

    with tf.variable_scope("loss"):
        '''
        fconv3_flat= tf.cast(fconv3, tf.float64)
        ground_truth = tf.cast(ground_truth, tf.float64)
        ground_truth = tf.expand_dims(ground_truth, axis=3)
        '''

        fconv3_flat = tf.reshape(fconv3, [-1])

        fconv3_flat = tf.Print(fconv3_flat, [fconv3_flat], message='fconv3_flat')
        depth_flat = tf.reshape(ground_truth, [-1])
        depth_flat = tf.Print(depth_flat, [depth_flat], message='depth_flat')

        '''
        d = tf.subtract(fconv3_flat, depth_flat)
        d = tf.Print(d,[d], message="value of d")
        square_d = tf.square(d)
        square_d = tf.Print(square_d, [square_d], message="value of square_d")
        sum_square_d = tf.reduce_sum(square_d, 1)
        sum_square_d = tf.Print(sum_square_d, [sum_square_d], message="value of sum_square_d")
        sum_d = tf.reduce_sum(d, 1)
        sum_d = tf.Print(sum_d, [sum_d], message="value of sum_d")
        square_sum_d = tf.square(sum_d)
        square_sum_d = tf.Print(square_sum_d, [square_sum_d], message="value of square_sum_d")


        cost = tf.reduce_mean(sum_square_d / 256.0 * 512.0 - 0.5 * square_sum_d /tf.cast(tf.pow(256 * 512, 2), dtype = tf.float32))
        cost = tf.Print(cost,[cost], message="cost value")
        '''

        # cov, op = tf.contrib.metrics.streaming_covariance(fconv3,ground_truth)

        fconv3 = tf.Print(fconv3, [tf.reduce_max(fconv3), tf.reduce_min(fconv3)], message="fconv3 max and min")

        fconv = tf.div(
            tf.subtract(
                fconv3_flat,
                tf.reduce_min(fconv3_flat)
            ),
            tf.subtract(
                tf.reduce_max(fconv3_flat),
                tf.reduce_min(fconv3_flat)
            )
        )

        ground_truth = tf.Print(ground_truth, [tf.reduce_max(ground_truth), tf.reduce_min(ground_truth)],
                                message='grounftruth max and min')

        fconv = tf.Print(fconv, [tf.reduce_max(fconv), tf.reduce_min(fconv)], message='fconv max and min')
        # fconv_norm = abs(tf.norm(fconv))
        # fconv_norm = tf.Print(fconv_norm, [fconv_norm], message='fconv_norm')
        # fconv = cv2.normalize(fconv1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        min_depth = np.min(depth_flat)
        print('Min_depth', min_depth)
        max_depth = np.max(depth_flat)
        print('Max_depth', max_depth)

        er_matA = abs(fconv - depth_flat)
        err_mat = tf.Print(er_matA, [tf.reduce_max(er_matA), tf.reduce_min(er_matA)],
                           message="err_matrix max and min")

        er_Norm = tf.norm(err_mat)
        er_Norm = tf.Print(er_Norm, [er_Norm], message='error_norm')
        # num_image = np.shape(image)[0]
        min_Norm = np.zeros((8, 256, 512, 1), dtype=np.float32)
        max_Norm = np.linalg.norm(np.ones((8, 256, 512, 1), dtype=np.float32))
        max_Norm = tf.Print(max_Norm, [max_Norm], message='max_Norm')

        acc = (er_Norm / max_Norm) * 100
        accuracy = 100 - acc
        # accuracy = tf.reduce_mean(accuracy)
        accuracy = tf.Print(accuracy, [accuracy], message='accuracy')




        # SSIM
        ssim = tf.reduce_mean(tf.image.ssim(fconv3, ground_truth, 1.0))
        ssim = tf.Print(ssim, [ssim], message="SSIM VALUE")

        ssim = tf.where(tf.is_nan(ssim), 0., ssim)

        if ssim is not None:
            ssim = (ssim + 1) / 2


        else:
            ssim = 1e-5
            ssim = (ssim + 1) / 2


        acc_ssim = tf.Print(ssim,[ssim*100], message="Accuracy ssim")
        dssim = 1 - acc_ssim
        dssim = tf.Print(dssim, [dssim], message="D-SSIM")

        mse = tf.reduce_mean(tf.losses.mean_squared_error(fconv3, ground_truth))
        mse = tf.Print(mse, [mse], message="mse")

        # psnr = 1 - tf.reduce_mean(tf.image.psnr(fconv3,ground_truth,max_val=1.0))
        # psnr = tf.Print(psnr, [psnr], message="psnr")

        total_cost = dssim

        tf.add_to_collections('losses', total_cost)
        tf.add_to_collections('accuracy', accuracy)
        # bat_ac = tf.reduce_sum((1 - d))
        # return  tf.add_n(tf.get_collection('accuracy')),tf.add_n(tf.get_collection('losses'))

        return accuracy, dssim,acc_ssim
    '''
        fconv3 = tf.Print(fconv3, [tf.shape(fconv3)], message='fconv3')

        reg_const = 0.5
        log_y = tf.log(fconv3) - tf.log(ground_truth)
        loss_l1 = (log_y) ** 2

        penal = tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(loss_l1)))

        SI = loss_l1 - reg_const * penal
        loss = tf.reduce_mean(SI)

        loss = tf.reduce_mean(tf.losses.mean_pairwise_squared_error(fconv3, ground_truth))

        return loss
        #loss = tf.reduce_sum(tf.losses.absolute_difference(ground_truth, fconv3))
        #return loss



    '''