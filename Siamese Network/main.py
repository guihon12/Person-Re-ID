import tensorflow as tf
import numpy as np
import cv2
import utils
import csv
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '20', 'batch size for training')
tf.flags.DEFINE_integer('test_size', '10', 'batch size for testing')
tf.flags.DEFINE_integer('max_steps', '25000000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'weights/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, rank')

IMAGE_WIDTH = 140
IMAGE_HEIGHT = 90
# network2 - (upper)width : height = 140 : 90 / (person) : 100 : 120
# network - width : height = 160: 100

def preprocess(images, is_train):
    def train():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1])
        shape = [1 for _ in range(split[0].get_shape()[1])]
        for i in range(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in range(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)


def grid(feature):
    trans = tf.transpose(feature, [0, 3, 1, 2])
    shape = trans.get_shape().as_list()
    m1s = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
    reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
    result = tf.multiply(reshape, m1s)
    return result


def neighborhood(feature):
    trans = tf.transpose(feature, [0, 3, 1, 2])
    shape = trans.get_shape().as_list()
    reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
    result = []
    pad = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
    for i in range(shape[2]):
        for j in range(shape[3]):
            result.append(pad[:, :, :, i:i + 5, j:j + 5])
    concat = tf.concat(result, axis=0)
    reshape = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
    result = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
    return result


def cross(feature1, feature2):
    trans = tf.transpose(feature1, [0, 3, 1, 2])
    shape = trans.get_shape().as_list()
    f = grid(feature1)
    g = neighborhood(feature2)
    reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
    f = grid(feature2)
    g = neighborhood(feature1)
    reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
    return reshape1, reshape2


def network(images1, images2, weight_decay):
    with tf.variable_scope('network'):
        # Tied Convolution
        conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
        pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
        conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
        pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
        conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
        pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
        conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
        pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

        # Cross-Input Neighborhood Differences
        reshape1, reshape2 = cross(pool1_2, pool2_2)

        k1 = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name='k1')
        k2 = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name='k2')

        # Patch Summary Features
        l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
        l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

        # Across-Patch Features
        m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
        pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
        m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
        pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

        # Higher-Order Relationships
        concat = tf.concat([pool_m1, pool_m2], axis=3)
        reshape = tf.reshape(concat, [FLAGS.batch_size, -1])
        fc1 = tf.layers.dense(reshape, 500, tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, 2, name='fc2')
        return fc2


def training(batch_size, height, width, lr):
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    images = tf.placeholder(tf.float32, [2, batch_size, height, width, 3], name='images')
    labels = tf.placeholder(tf.float32, [batch_size, 2], name='labels')
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    num_id = utils.get_num_id(FLAGS.data_dir, 'train')
    image1, image2 = preprocess(images, is_train)
    print('Build network')
    logits = network(image1, image2, weight_decay)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    train = optimizer.minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            saver.restore(sess, ckpt.model_checkpoint_path)
        step = sess.run(global_step)
        for i in range(step, FLAGS.max_steps + 1):
            batch_images, batch_labels = utils.read_data(FLAGS.data_dir, 'train', num_id, width, height, batch_size)
            feed_dict = {learning_rate: lr, images: batch_images, labels: batch_labels, is_train: True}
            sess.run(train, feed_dict=feed_dict)
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print('Step: %d, Learning rate: %f, Train loss_p: %f' % (i, lr, train_loss))
            lr = FLAGS.learning_rate * ((0.01 * i + 1) ** -0.75)
            if i % 1000 == 0:
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)


def cal_rank(batch_size, height, width, num_rank):
    images = tf.placeholder(tf.float32, [2, batch_size, height, width, 3], name='images')
    labels = tf.placeholder(tf.float32, [batch_size, 2], name='labels')
    is_train = tf.placeholder(tf.bool, name='is_train')
    weight_decay = 0.0005
    image1, image2 = preprocess(images, is_train)
    print('Build network')
    logits = network(image1, image2, weight_decay)
    file_path = '%s/val/' % (FLAGS.data_dir)
    query = os.listdir(file_path)
    rank = np.zeros(num_rank)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            saver.restore(sess, ckpt.model_checkpoint_path)
            #saver.restore(sess, './logs/model.ckpt-79500')
        for i in range(len(query)):
            gallery = utils.gallery_make(query, i)
            batch_features, batch_labels = utils.multi_batch_data(FLAGS.data_dir, query, gallery, i, height,
                                                                  width)
            temp = []
            for j in range(len(gallery)):
                feed_dict = {images: batch_features[:, j].reshape(2, 1, height, width),
                             labels: batch_labels[j].reshape(1, 2), is_train: False}
                score = sess.run(logits, feed_dict=feed_dict)
                score = score.reshape(2)[0]
                temp.append(score)
            print(i, 'images done')
            # num_rank(10) 랭킹 리스트의 인덱스 리스트
            index = np.flip(np.argsort(temp)[-(num_rank + 1):])
            print(query[i])
            print(temp[index[0]])
            for j in range(num_rank):
                print(gallery[index[j]])
                if query[i].split('_')[0] == gallery[index[j]].split('_')[0]:
                    rank[j:] = rank[j:] + np.ones(num_rank - j)
                    break
            for j in range(num_rank):
                print(rank[j] / (i + 1))



def main(argv=None):
    if FLAGS.mode == 'train':
        training(FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, FLAGS.learning_rate)
    elif FLAGS.mode =='cal_rank':
        cal_rank(FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 10)


if __name__ == '__main__':
    tf.app.run()