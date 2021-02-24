# -*- encoding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import trinet
import os
import sys
from tensorflow.python.ops import data_flow_ops


def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                                       dtypes=[tf.string, tf.int32, tf.int32],
                                                       shapes=[(1,), (1,), (1,)],
                                                       shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many(
                [image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = trinet.create_input_pipeline(eval_input_queue, image_size,
                                                                     nrof_preprocess_threads,
                                                                     batch_size_placeholder)
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            sess.run(tf.global_variables_initializer())
            dir = os.listdir(args.model_dir)
            for filename in dir:
                ext = os.path.splitext(filename)[-1]
                if ext == '.meta':
                    meta_file = args.model_dir + filename
                elif ext == '.index':
                    model = args.model_dir + filename.replace(".index", "")
            print('restoring')
            saver = tf.train.import_meta_graph(meta_file, input_map=input_map)
            sess.run(tf.global_variables_initializer())
            saver.restore(tf.get_default_session(), model)
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            print('restoring success')
            generate_emb(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                         batch_size_placeholder, control_placeholder, embeddings, args.data_dir)


def generate_emb(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                 batch_size_placeholder, control_placeholder, embeddings, mtcnn_img_dir):
    labels_array = np.expand_dims(np.arange(0, 1), 1)
    control_array = np.zeros_like(labels_array, np.int32)
    emb_save_dir = '../data/npy/' + mtcnn_img_dir.split('/')[-1] + '/'
    if not os.path.exists(emb_save_dir):
        os.makedirs(emb_save_dir)
    folder_img_name = [s for s in os.listdir(mtcnn_img_dir) if not ".txt" in s]
    all_img_name = []
    count_img = []
    count = 0
    for name in folder_img_name:
        img_dir = mtcnn_img_dir + '/' + name
        count_img.append(count)
        for i in range(0, len(os.listdir(img_dir))):
            all_img_name.append([n for n in os.listdir(img_dir)][i])
            count = count + 1
    print(len(all_img_name))
    for i in range(0, len(all_img_name)):
        if i in count_img:
            middle_name = folder_img_name[count_img.index(i)]
        image_path = mtcnn_img_dir + '/' + middle_name + '/' + all_img_name[i]
        image_path = np.reshape(np.array(image_path), [1, 1])
        sess.run(enqueue_op, {image_paths_placeholder: image_path, labels_placeholder: labels_array,
                              control_placeholder: control_array})
        feed_dict = {phase_train_placeholder: False, batch_size_placeholder: 1}
        emb = sess.run([embeddings], feed_dict=feed_dict)
        sys.stdout.flush()
        if not os.path.exists(emb_save_dir + middle_name):
            os.makedirs(emb_save_dir + middle_name)
        emb_save_name = emb_save_dir + middle_name + '/' + all_img_name[i].split('.jpg')[0]
        np.save(emb_save_name, emb)
        # print(np.shape(emb)) # 8x8x1088
        print(emb_save_name)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to the data directory', default='../data/224/market1501/query')
    parser.add_argument('--model_dir', type=str, help='the meta_file and ckpt_file', default='../weight/')
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=224)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))