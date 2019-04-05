from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import cv2
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os.path
from sklearn.preprocessing import Normalizer
import PIL
from sklearn.cross_validation import KFold
from tqdm import tqdm
import model
import preprocess
from sklearn.model_selection import train_test_split
from tensorflow.python.client import timeline
import depth_utils


checkpoint_path = '/home/angel/monodepth/chkpoint'


tf.reset_default_graph()


chk_image = '/home/angel/2011_09_26/DepthEstimation/TopViewRGB_Images/image_00930_0.png'

PLOT_DIR = '/home/angel/monodepth/OP_plots/plots'

PLOT_DIR = '/home/angel/monodepth/OP_plots/plots'

#is_training = True







def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    depth_utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    print('shape of conv weights:', np.shape(weights))
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = depth_utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder


    print("inside PLOT CONV OUTPUT")
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    depth_utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)





    if len(np.shape(conv_img)) is 3:
        print('shape len is 3')
        conv_img = np.expand_dims(conv_img, axis= 0)
        print('num_dims_shape:', conv_img.shape)
        num_filters = conv_img.shape[3]
        print('num filters:', num_filters)
    # get number of convolutional filters

    else:
        print('else num filters')
        num_filters = conv_img.shape[3]
        print('num_filters', num_filters)
    # get number of grid rows and columns

    if num_filters is 1:
        grid_r,grid_c = 1,1
        print('grid_r,grid_c', grid_r,grid_c)

    else:
        print("inside else")
        grid_r, grid_c = depth_utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))



    if num_filters is 1:
        print('inside num_filters 1')
        conv_img = np.squeeze(conv_img,axis=3)
        print('shape of conv image filter 1:', np.shape(conv_img))
        img = conv_img[0, :, :]
        # put it on the grid
        axes.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        axes.set_xticks([])
        axes.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


    else:


        # iterate filters
        for l, ax in enumerate(axes.flat):
            print('l,ax:', l,ax)


            # get a single image
            #img = conv_img[0, :, :, l]
            img = conv_img[0, :, :, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
            # save figure
            plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


'''
def extracting(meta_dir):
    num_tensor = 0
    var_name = ['2-convolutional/kernel']
    model_name = meta_dir
    configfiles = [os.path.join(dirpath, f)  # List of META files
    for dirpath, dirnames, files in os.walk(model_name)
    for f in fnmatch.filter(files, '*.meta')]

    with tf.Session() as sess:
        try:
            # A MetaGraph contains both a TensorFlow GraphDef
            # as well as associated metadata necessary
            # for running computation in a graph when crossing a process boundary.
            saver = tf.train.import_meta_graph(configfiles[0])
       except:
           print("Unexpected error:", sys.exc_info()[0])
       else:
           # It will get the latest check point in the directory
           saver.restore(sess, configfiles[-1].split('.')[0])  # Specific spot

           # Now, let's access and create placeholders variables and
           # create feed-dict to feed new data
           graph = tf.get_default_graph()
           inside_list = [n.name for n in graph.as_graph_def().node]

           print('Step: ', configfiles[-1])

           print('Tensor:', var_name[0] + ':0')
           w2 = graph.get_tensor_by_name(var_name[0] + ':0')
           print('Tensor shape: ', w2.get_shape())
           print('Tensor value: ', sess.run(w2))
           w2_saved = sess.run(w2)  # print out tensor

'''

def average_gradients(tower_grads):
    print("Average gradients")


    average_grads = []

    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        #print(grad_and_vars)
        grads = []
        for g, v in (grad_and_vars):
            # Add 0 dimension to the gradients to represent the tower.
            #print("grad:",g)
            #print("var:",v)

            if g is not None:


                expanded_g = tf.expand_dims(g, 0)

            else:

                expanded_g = tf.zeros_like(v)
                expanded_g = tf.expand_dims(expanded_g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)



        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        average_grads.append(grad_and_var)




    vars = [x[1] for x in average_grads]
    print(vars)
    gradients = [x[0] for x in average_grads]
    print(gradients)



    clipped = [tf.clip_by_norm(g, 2.5) for g in gradients]

    grads_vars = zip(clipped, vars)

    return grads_vars




def _add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    print(type(loss_averages_op))
    return loss_averages_op


xtrain, ytrain, xval, yval, xtest, ytest = preprocess.data_process()
#x_train = tf.convert_to_tensor(xtrain)#xtrain = np.expand_dims(xtrain,3)
#xtrain = np.expand_dims(xtrain,3)
#ytrain = np.expand_dims(ytrain,3)

#xval = np.expand_dims(xval,3)
#yval = np.expand_dims(yval,3)
#xtest = np.expand_dims(xtest,3)
#ytest = np.expand_dims(ytest,3)
'''
x_train = tf.expand_dims(tf.convert_to_tensor(xtrain),3)
y_train = tf.expand_dims(tf.convert_to_tensor(ytrain),3)


x_val = tf.expand_dims(tf.convert_to_tensor(xval),3)
y_val = tf.expand_dims(tf.convert_to_tensor(yval),3)

x_test = tf.expand_dims(tf.convert_to_tensor(xtest),3)
y_test = tf.expand_dims(tf.convert_to_tensor(ytest),3)

'''
print(np.shape(xtrain))
print(xtrain.dtype)

print(np.shape(xval))
print(np.shape(yval))

shape = np.shape(xtrain)
num_samples_train = shape[0]
print("num of samples train:", num_samples_train)
shape2 = np.shape(xval)
num_samples_val  = shape2[0]
print("num of samples val:", num_samples_val)
shape3 = np.shape(xtest)
num_samples_test = shape3[0]
print("num of samples  test:", num_samples_test)



def training():






    with tf.device('/device:GPU:0'):



        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)







        batch_size = 8
        num_epochs = 20
        repeat = 5
        steps_per_epoch = np.ceil((num_samples_train *repeat)/ batch_size).astype(np.int32)
        print('steps_per_epoch_train', steps_per_epoch)
        num_total_steps = steps_per_epoch * num_epochs
        print('total_num_steps:', num_total_steps)

        steps_per_epoch_val = np.ceil((num_samples_val*repeat) / batch_size).astype(np.int32)
        print('steps_per_epoch_val', steps_per_epoch_val)
        #num_total_steps = steps_per_epoch * num_epochs
        #print('total_num_steps:', num_total_steps)
        steps_per_epoch_test = np.ceil(num_samples_test / batch_size).astype(np.int32)
        print('steps_per_epoch_test', steps_per_epoch_test)


        learning_rate_init = 1e-4
        boundaries = [np.int32((3 / 5) * num_total_steps), np.int32((4 / 5) * num_total_steps)]
        print("boundaries:{}".format(boundaries))
        values = [learning_rate_init, learning_rate_init / 2, learning_rate_init / 4]
        print("values:{}".format(values))
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        print("learning rate:{}".format(learning_rate))
        opt_step = tf.train.AdamOptimizer(learning_rate)

        #x_init = tf.placeholder(tf.float32,shape = (5,256,512,3))
        #y_init = tf.placeholder(tf.float32, shape =(5,256,512,1))
        #vf_init = tf.placeholder(tf.float32, shape=(160, 256, 512, 3))
        #vg_init = tf.placeholder(tf.float32, shape=(160, 256, 512, 1))


        #x_train = tf.Variable(x_init,trainable= False)
        #y_train =  tf.Variable(y_init,trainable= False)
        #x_val = tf.Variable(vf_init,trainable= False)

        #y_val = tf.Variable(vg_init,trainable=False)


        '''
        dataset1 = tf.data.Dataset.from_tensor_slices((X_train)
        print(dataset1.output_shapes)
        print(dataset1.output_types)
        print(np.shape(X_train))

        dataset2 = tf.data.Dataset.from_tensor_slices(Y_train)
        print(dataset2.output_shapes)
        print(dataset2.output_types)
        '''

        #is_training = tf.placeholder(tf.bool, name="is_training")


        placeholder_X = tf.placeholder(tf.float32,shape = [None,256,512,3],name='X_train')
        #var_X = tf.Variable(placeholder_X, dtype=tf.float32, trainable=False)
        placeholder_Y = tf.placeholder(tf.float32, shape = [None,256,512,1], name = 'Y_train')
        #var_Y = tf.Variable(placeholder_Y,dtype=tf.float32,trainable=False)


        #val_X = tf.placeholder(tf.float32,shape=[160,256,512,3],name='X_Val')
        #v_x = tf.Variable(val_X,dtype=tf.float32,  trainable=False)
        #val_Y = tf.placeholder(tf.float32, shape=[160, 256, 512, 1],name='Y_Val')
        #v_y = tf.Variable(val_Y,dtype=tf.float32, trainable=False)

        #Create seperate datasets for Training , Validation and Testing datsets

        train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,placeholder_Y))
        #train_dataset = tf.data.Dataset.zip((dataset1, dataset2)).shuffle(100).repeat().batch(8)
        train_dataset = train_dataset.shuffle(buffer_size = 100).repeat(5).batch(8)

        val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,placeholder_Y))
        val_dataset = val_dataset.shuffle(buffer_size = 100).repeat(5).batch(8)

        test_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,placeholder_Y))
        test_dataset = test_dataset.batch(8)
        #Create handle for the iterator



        #handle = tf.placeholder(tf.string,  shape = [], name='str_hand')
        #iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        #data_X, data_Y = iterator.get_next()

        # Create Reinitializable iterator for Train and Validation, one shot iterator for Test
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        #iterator = train_dataset.make_one_shot_iterator()
        data_X, data_Y = iterator.get_next()

        train_iterator_init_op = iterator.make_initializer(train_dataset)
        #val_iterator = tf.data.Iterator.from_structure(val_dataset.output_types, val_dataset.output_shapes)

        val_iterator_init_op = iterator.make_initializer(val_dataset)
        test_iterator_init_op= iterator.make_initializer(test_dataset)

        #ssim_loss = tf.placeholder(tf.float64, name='ssim')
        #mse_loss = tf.placeholder(tf.float64, name = 'mse')
        #psnr_loss = tf.placeholder(tf.float64, name = 'psnr')

        #acc = tf.placeholder(tf.float64, name = 'accuracy')



        tower_loss = []
        tower_grads = []
        tower_acc = []
        tower_acsim = []
        num_gpus = 2

        # SESSION
        gpu_fraction = 0.5
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        sess = tf.Session(config=config)
        # 3run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('training') as cur_scope:
            print("inside training loop")
            for i in range(num_gpus):
                with tf.device('/device:GPU:%d' % i):


                    acc, loss,ac_sim = model.vgg(data_X, data_Y)
                    loss = tf.Print(loss,[loss],message="returned loss value")
                    tower_acc.append(acc)
                    #loss_avg = _add_loss_summaries(loss)

                    #loss_avg = tf.Print(loss_avg, [loss_avg], message='LOSS')
                    tower_loss.append(loss)
                    tower_acsim.append(ac_sim)
                    reuse_variables = True
                    cur_scope.reuse_variables()
                    #var_list = tf.trainable_variables()
                    grads = opt_step.compute_gradients(loss)




                    tower_grads.append(grads)




        grads_vars = average_gradients(tower_grads)
        #grads, _ = tf.clip_by_norm(grads_vars,2.5)
        #tvars = tf.trainable_variables()
        bat_ac = tf.reduce_mean(tower_acc,axis=0)
        bat_ac_sim = tf.reduce_mean(tower_acsim,axis=0)
        val_loss = tf.reduce_mean(tower_loss,axis = 0)
        with tf.control_dependencies(update_ops):
           apply_gradient_op = opt_step.apply_gradients(grads_vars, global_step=global_step)
        #total_loss = tf.reduce_mean(tower_loss)
        #total_loss = tf.Print(total_loss,  [total_loss], message="TOTAL LOSS")
        tf.summary.scalar('learning_rate', learning_rate, ['model_0'])

        #val_loss =tf.get_collection('losses')
        #val_loss = tf.Print(val_loss, [val_loss], message="val_loss")

        tf.summary.scalar('total_loss', val_loss, ['model_0'])
        tf.summary.scalar('accuracy', bat_ac,['model_0'])


        summary_op = tf.summary.merge_all('model_0')


        summary_writer = tf.summary.FileWriter('./newsum', sess.graph)
        train_saver = tf.train.Saver()


        # COUNT PARAMS
        total_num_parameters = 0
        #print('trainable var', tf.trainable_variables)
        for variable in tf.trainable_variables():
            e_var = np.array(variable.get_shape().as_list()).prod()
            #print(variable.name, e_var)
            total_num_parameters += e_var
        print("number of trainable parameters: {}".format(total_num_parameters))

        # GO!
       # sess.graph.finalize()
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #train_val_string = sess.run(train_val_iterator.string_handle())
        #val_string = sess.run(val_iterator.string_handle())
        #val_string = sess.run(train_val_iterator.string_handle())
        #sess.run(train_iterator_op)


        #print(x_train,y_train)
        start_step = global_step.eval(session=sess)
        print('start_step:', start_step)
        start_time = time.time()
        print("start_time")



        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        #sess.graph.finalize()

        loss_list_train = []
        acc_list_train = []
        acc_train_sim = []
        loss_list_val = []
        acc_list_val = []
        acc_sim_val = []




        for epoch  in range(start_step,num_epochs):

            train_loss = 0

            train_accuracy = 0
            validation_loss = 0
            validation_accuracy = 0
            train_acc_sim = 0
            val_acc_sim = 0
            print('iteration starts')
            print('epoch_num:', epoch)
            is_training = True

            sess.run(train_iterator_init_op, feed_dict={placeholder_X: xtrain,
                                                placeholder_Y:ytrain})



            #options = run_options, run_metadata = run_metadata, add this in the feed

            try:


                with tqdm(total=len(ytrain)*repeat) as pbar:

                    while True:


                        #before_op_time = time.time()
                        # Feed to feedable iterator the string handle of reinitializable iterator
                        #handle_string = sess.run([train_val_string])
                        _, acc,loss_value,ac_sim = sess.run([apply_gradient_op,bat_ac, val_loss,bat_ac_sim])

                        train_loss += loss_value
                        train_accuracy += acc
                        train_acc_sim += ac_sim
                        #duration = time.time() - before_op_times

                        '''
                        examples_per_sec = batch_size / duration
                        time_sofar = (time.time() - start_time) / 3600
                        training_time_left = (num_total_steps / step - 1.0) * time_sofar
                        print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                        print(print_string.format(step, examples_per_sec, train_loss / (step + 0.00001), time_sofar, training_time_left))
                        '''

                        '''
                        #if step and step % 5 == 0:
                        #train_saver.save(sess, './model-vgg/mod1ckpt.ckpt', global_step=step)
                        

                        conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={placeholder_X: xtrain[:1]})
                        print('conv_out_shape:', np.shape(conv_out))

                        print('conv_out_shape:', np.shape(conv_out))
                        # conv_out = sess.run([tf.get_collection('conv_output')])
                        for i, c in enumerate(conv_out[0]):
                            plot_conv_output(c, 'conv{}'.format(i))
                        '''
                        pbar.update(batch_size)

            except tf.errors.OutOfRangeError:

                print('training epoch complted')

            '''

            if epoch and epoch % 5 == 0:
                step = (steps_per_epoch + steps_per_epoch_val) * epoch
                print('stepno:', step)
                print('checküpointing')

                train_saver.save(sess, './model-vgg/vgg.ckpt', global_step=epoch, write_meta_graph=True)

            

            if epoch and epoch % 5 == 0:
                step = (steps_per_epoch + steps_per_epoch_val ) * epoch
                print('stepno:',step)
                print('summarizing')
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, global_step = step)


             
                train_saver.save(sess, './model-vgg/vgg.ckpt', global_step=epoch, write_meta_graph= False)

            

                
                print('step no:', step)
                before_op_time = time.time()
                _, acc, loss_value = sess.run([apply_gradient_op,bat_ac, total_loss], feed_dict= {handle:train_string})
                duration = time.time() - before_op_time
                train_loss += loss_valuereturned loss value[0.832219243]
 
                train_loss /=(step +1)
                train_accuracy +=acc
                #train_loss = tf.Print(train_loss,[train_loss], message="traing loss")
                train_accuracy /= (step + 1)
                #train_accuracy = tf.Print(train_accuracy,[train_accuracy], message="train ccuracy")
                print(loss_value)
                print(acc)
                # tf.verify_tensor_all_finite(loss_value,'tensor finite')

                print("print duration", duration)

                #if step and step % 1 == 0:
                #print('inside in')
                    examples_per_sec = batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    num_total_steps = num_epochs * steps_per_epoch
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar
                    print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | acc: {: .5f} |time elapsed: {:.2f}h | time left: {:.2f}h'
                    print(print_string.format(step, examples_per_sec, train_loss, train_accuracy, time_sofar,
                                                          training_time_left))
                '''

                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str, global_step=step)

            #print('\n Epoch: {}'.format(epoch + 1))

            #training_loss = train_loss
            #print(training_loss)
            train_loss = train_loss / steps_per_epoch
            print("Training_Loss:", train_loss)

            loss_list_train.append(train_loss)
            #training_loss = tf.Print(training_loss,[training_loss], message="train loss")
            #print('Training loss: {:.4f}'.format(training_loss))
            #print('Training acc: {:.4f}'.format(train_accuracy / steps_per_epoch))

            train_accuracy = (train_accuracy / steps_per_epoch)

            print("Training_ACC:", train_accuracy)


            acc_list_train.append(train_accuracy)

            train_acc_sim = (train_acc_sim / steps_per_epoch)

            print("Training_ACC_SIM:", train_acc_sim)

            acc_train_sim.append(train_acc_sim)

            sess.run(val_iterator_init_op, feed_dict={placeholder_X: xval, placeholder_Y:yval})

            try:
                while True:
                    print('validating')
                    _ ,v_acc, v_loss,ac_sim = sess.run([apply_gradient_op,bat_ac, val_loss,bat_ac_sim])
                    validation_loss += v_loss
                    validation_accuracy += v_acc
                    val_acc_sim += ac_sim

            except tf.errors.OutOfRangeError:
                print('validation epoch complted')




            validation_loss = validation_loss / steps_per_epoch_val
            print('validation_loss:', validation_loss)
            loss_list_val.append(validation_loss)

            #print('Val loss: {:.4f}\n'.format(validation_loss / steps_per_epoch_val))
            #print('Val acc: {:.4f}\n'.format(validation_accuracy /steps_per_epoch_val))
            validation_accuracy = (validation_accuracy / steps_per_epoch_val)
            print('validation_accuracy:', validation_accuracy)
            acc_list_val.append(validation_accuracy)
            #tf.summary.scalar('total_loss', loss_value, ['model_0'])

            val_acc_sim = (val_acc_sim / steps_per_epoch_val)
            print('val_ACC_SIM:', val_acc_sim)
            acc_sim_val.append(val_acc_sim)
            '''
            conv_weights = sess.run([tf.get_collection('conv_weights')])
            print('conv_weights_shape', np.shape(conv_weights))
            print('shape of convweightss[0]:', np.shape(conv_weights[0]))

            for i, c in enumerate(conv_weights[0]):
                plot_conv_weights(c, 'conv{}'.format(i))
            '''
            # get output of all convolutional layers
            # here we need to provide an input image
            




        # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        # retrain = False



        #ip_image = sess.run([tf.get_collection('input_images')])
        #for i, c in enumerate(ip_image[0]):
       # plot_conv_output(c, 'im{}'.format(i))

        '''
        saver = tf.train.import_meta_graph('model-vgg-1000.meta')
        train_saver.restore(sess, '/home/angel/monodepth/chkpoint.ckpt')
        
        
    
        '''
        print("train accuracy:", acc_list_train)
        print("train accuracy SIM:", acc_train_sim)
        print("train loss:", loss_list_train)
        print("val accuracy:", acc_list_val)
        print("val accuracy:", acc_sim_val)
        print("val loss:", loss_list_val)




        loss_list_test = []
        acc_list_test = []
        acc_sim_test =[]
        test_loss, test_accuracy,test_sim_ac = 0, 0, 0
        is_training = False
        sess.run(test_iterator_init_op, feed_dict={placeholder_X: xtest, placeholder_Y: ytest})
        try:
            while True:
                print("Testing")
                # Feed to feedable iterator the string handle of one shot iterator
                loss, acc,ac_sim,conv_out = sess.run([val_loss,bat_ac,bat_ac_sim,tf.get_collection('conv_output')])
                test_loss += loss
                test_accuracy += acc
                test_sim_ac += ac_sim

                print('conv_out shape:', np.shape(conv_out[0]))
                for i, c in enumerate(conv_out):
                    plot_conv_output(c, 'conv{}'.format(i))


        except tf.errors.OutOfRangeError:
            print('Testing epoch complted')
            pass




        #print('\nTest accuracy: {:.4f}, loss: {:.4f}'.format(test_accuracy / len(y_test), test_loss / len(y_test)))
        test_loss = test_loss/steps_per_epoch_test
        print('Test loss:',test_loss)

        loss_list_test.append(test_loss)

        # print('Val loss: {:.4f}\n'.format(validation_loss / steps_per_epoch_val))
        # print('Val acc: {:.4f}\n'.format(validation_accuracy /steps_per_epoch_val))
        test_acc = test_accuracy / steps_per_epoch_test
        print('Test accuracy:',test_acc)
        #sess.run(test_iterator.initializer, feed_dict={placeholder_X: xtest, placeholder_Y: ytest, is_training: False})
        #conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={placeholder_X: xtest[:1]})
        acc_list_test.append(test_acc)

        test_sim = (test_sim_ac / steps_per_epoch_test)
        print('Test accuracy_sim:',test_sim)

        acc_sim_test.append(test_sim)



        print("Test Loss:", loss_list_test)
        print("Test Acc:",acc_list_test)
        print("Test ACC_SIM:", acc_sim_test)
    sess.close()

    '''
    if retrain:
        sess.run(global_step.assign(0))
    '''

    summary_writer.flush()
    summary_writer.close()
    #tf.contrib.keras.backend.clear_session()






def main(_):
    #print("inside main")
    mode = 'train'



    if mode == 'train':
        training()
    elif mode == 'test':
        return

if __name__ == '__main__':
    #print("inside app run")
    tf.app.run()
