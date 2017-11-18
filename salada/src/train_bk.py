# -*- coding: utf-8 -*- 
import tensorflow as tf
import numpy as np
import configparser
import os
from enum import Enum
import cv2
import img_common
import math


class Param_check(Enum):
    EXIST_PATH = 0



def set_config(conf):
    config = configparser.ConfigParser()
    config.sections() # [section]
    config.read(conf)
    return config


def check_value(value, chk_kbn):
    """ check parameter
        kbn
          0:path existing check

    """

def data_loader(data_list_file, data_dir, resize_size, number_of_class):
    image_list = []
    label_list = []

    with open(data_list_file, 'r') as train_txt:
        for i, line in enumerate(train_txt):
            line = line.rstrip()
            vals = line.split(' ')
            image_path = data_dir + vals[0]
            image = cv2.imread(image_path)
            image = img_common.resize_image(image, resize_size)
            # image.flatten() mean 1 dimension and normalize
            image_list.append(image.flatten().astype(np.float32) / 255.0)
            tmp = np.zeros(number_of_class)
            tmp[int(vals[1])] = 1
            label_list.append(tmp)
        images = np.asarray(image_list)
        labels = np.asarray(label_list)

    return images, labels


def inference(image_placeholder, keep_prob, resize_size, number_of_classes):
    def weight_variable(shape, num):
        # truncated_normal: Generate value of normalize using Gaussian distribution(stddev * 2)
        initial = tf.truncated_normal(shape, stddev=0.1, mean=0.)
        return (tf.Variable(initial).initialized_value())

    def bias_variable(shape):
        # constant: Generate value of constant
        initial = tf.constant(0.0, shape=shape)
        return (tf.Variable(initial).initialized_value())

    def conv2d_first(x, W):
        # conv2d: 1st aug = 4 dimension tensor[batch size, height, width, channels]
        #         2nd aug = weight
        #         3rd aug = slide size of kernel
        #         4th aug = padding way if you want to zero padding then set 'SAME'
        return (tf.nn.conv2d(x, W, strides=[1, 4, 4, 1], padding='SAME'))

    def conv2d(x, W):
        return (tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'))

    def max_pool_3x3(x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(image_placeholder, [-1, resize_size[0], resize_size[1], 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([11, 11, 3, 96], resize_size[0] * resize_size[1])
        b_conv1 = bias_variable([96])
        h_conv1 = tf.nn.relu(conv2d_first(x_image, W_conv1 + b_conv1))
        print(h_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_3x3(tf.nn.local_response_normalization(h_conv1))
        print(h_pool1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 96, 256], 96)
        b_conv2 = bias_variable([256])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2 + b_conv2))
        print(h_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_3x3(tf.nn.local_response_normalization(h_conv2))
        print(h_pool2)

    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 256, 384], 256)
        b_conv3 = bias_variable([384])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        print(h_conv3)

    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([3, 3, 384, 384], 384)
        b_conv4 = bias_variable([384])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
        print(h_conv4)

    with tf.name_scope('conv5') as scope:
        W_conv5 = weight_variable([3, 3, 384, 256], 384)
        b_conv5 = bias_variable([256])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)
        print(h_conv5)

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_3x3(h_conv5)
        print(h_pool3)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*256, 4096], (7*7*256))
        b_fc1 = bias_variable([4096])
        h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        w_fc2 = weight_variable([4096,4096], 4096)
        b_fc2 = bias_variable([4096])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([4096, number_of_classes], 4096)
        b_fc3 = bias_variable([number_of_classes])
        # y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    return y_conv

def loss(logits, labels):
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0))))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

def main():
    # load configuration
    conf_file = '../conf/config.txt'
    config = set_config(conf_file)

    # set configuration
    param_chk = []
    root_dir = os.path.abspath(('..')) + '/'
    train_data = root_dir + config['data']['TRAIN_LIST_PATH']
    param_chk.append((train_data, Param_check.EXIST_PATH))
    test_data = root_dir + config['data']['TEST_LIST_PATH']
    param_chk.append((test_data, Param_check.EXIST_PATH))
    data_dir = root_dir + config['data']['DATA_PATH']
    log_dir = root_dir + 'log/'

    image_x_size = int(config['image']['IMAGE_SIZE'])
    image_y_size = int(config['image']['IMAGE_SIZE'])
    resize_size = (image_x_size, image_y_size)
    number_of_classes = int(config['image']['NUM_CLASSES'])
    image_pixels = image_x_size * image_y_size * 3
    learning_rate = float(config['hyper_parameter']['LEARNING_RATE'])
    max_step = int(config['hyper_parameter']['MAX_STEPS'])
    batch_size = int(config['hyper_parameter']['BATCH_SIZE'])

    # load train data and test data
    train_images, train_labels = data_loader(train_data, data_dir, resize_size, number_of_classes)
    test_images, test_labels = data_loader(test_data, data_dir, resize_size, number_of_classes)

    # hoge hoge
    with tf.Graph().as_default():
        images_placeholder = tf.placeholder('float32', shape=(None, image_pixels))
        labels_placeholder = tf.placeholder('float32', shape=(None, number_of_classes))
        keep_prob = tf.placeholder('float')

        logits = inference(images_placeholder, keep_prob, resize_size, number_of_classes)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, learning_rate)
        acc = accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        # generate events.out.tfevents.. log files
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph_def)

        for step in range(max_step):
            for i in range(int(len(train_images)/ batch_size)):
                batch = batch_size * i
                point = sess.run(train_op, feed_dict={
                    images_placeholder: train_images[batch:batch+batch_size],
                    labels_placeholder: train_labels[batch:batch+batch_size],
                    keep_prob: 0.5
                })

                train_accuracy = sess.run(acc, feed_dict={
                    images_placeholder: train_images,
                    labels_placeholder: train_labels,
                    keep_prob: 1.0
                })
                print('step %d, training accuracy %g' % (step, train_accuracy))

                summary_str = sess.run(summary_op, feed_dict={
                    images_placeholder: train_images,
                    labels_placeholder: train_labels,
                    keep_prob: 1.0
                })
                summary_writer.add_summary(summary_str, step)

        print ('test accuracy %g' % (sess.run(acc, feed_dict={
            images_placeholder: test_images,
            labels_placeholder: test_labels,
            keep_prob: 1.0
        })))

        save_path = saver.save(sess, 'model.ckpt')

    images_placeholder = tf.placeholder('float32', shape=(None, image_pixels))
    labels_placeholder = tf.placeholder('float32', shape=(None, number_of_classes))
    keep_prob = tf.placeholder('float32')
    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()
    saver = sess.run(tf.global_variables_initializer())
    hoge = sess.run(tf.global_variables_initializer())
    saver.restore(sess,"model.ckpt")

    for i in range(len(test_images)):
        hoge = np.argmax(logits.eval(feed_dict={images_placeholder: [test_images[i]],keep_prob: 1.0 })[0])
        print("%s"%hoge)


if __name__ == '__main__':
    main()