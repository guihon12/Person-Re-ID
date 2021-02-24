from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
import trinet
from tensorflow.python.ops import data_flow_ops

def main(args):
    # import Network
    print('Backbone : %s' % args.backbone)
    network = importlib.import_module('models.' + args.backbone)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    # create log, model file
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    # Write arguments to a text file
    trinet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    np.random.seed(seed=args.seed)
    # dataset : class_name & image_path
    train_set = trinet.get_dataset(args.data_dir)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    if args.continue_training_dir:
        print('Pre-trained model: %s' % os.path.expanduser(args.continue_training_dir))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])
        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                # pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')
        # Build the inference graph

        # prelogits, endpoints = network.inference(image_batch, args.keep_probability,
        prelogits, endpoints = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)

        #print('test14u2y487u1041290437u120')
        #print(endpoints['vgg_16/conv5/conv5_3']) #'vgg_16/conv5/conv5_3/Relu:0'
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        #vgg_feature = tf.nn.l2_normalize(endpoints['vgg_16/conv5/conv5_3'], 1, 1e-10, name='vgg')
        block6 = tf.nn.l2_normalize(endpoints['Mixed_5b'], 1, 1e-10, name='Mixed_5b')
        block6 = tf.nn.l2_normalize(endpoints['Mixed_6a'], 1, 1e-10, name='Mixed_6a')
        block7 = tf.nn.l2_normalize(endpoints['Mixed_7a'], 1, 1e-10, name='Mixed_7a')
        # tf.graph.get_tensor('block6:0')

        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = trinet.triplet_loss(anchor, positive, negative, args.alpha)
        
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = trinet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables())

        # Create a saver (load_ImageNet)
        if args.load_ImageNet and args.backbone == 'inception_resnet_v2':
            print('variable setting')
            trainable_vars = []
            exclusions = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
            for var in tf.trainable_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion.strip()):
                        break
                else:
                    trainable_vars.append(var)
            saver_ImageNet = tf.train.Saver(trainable_vars, max_to_keep=10000)

        elif args.load_ImageNet and args.backbone == 'vgg16':
            print('variable setting')
            trainable_vars = []
            exclusions = ['vgg_16/fc8']
            for var in tf.trainable_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion.strip()):
                        break
                else:
                    trainable_vars.append(var)
            saver_ImageNet = tf.train.Saver(trainable_vars, max_to_keep=10000)

        elif args.load_ImageNet and args.backbone == 'vgg19':
            print('variable setting')
            trainable_vars = []
            exclusions = ['vgg_19/fc8']
            for var in tf.trainable_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion.strip()):
                        break
                else:
                    trainable_vars.append(var)
            saver_ImageNet = tf.train.Saver(trainable_vars, max_to_keep=10000)

        elif args.load_ImageNet and args.backbone == 'resnet_v2_50':
            print('variable setting')
            trainable_vars = []
            exclusions = ['resnet_v2_50/logits', 'resnet_v2_50/fully_connected']
            for var in tf.trainable_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion.strip()):
                        break
                else:
                    trainable_vars.append(var)
            saver_ImageNet = tf.train.Saver(trainable_vars, max_to_keep=10000)

        elif args.load_ImageNet and args.backbone == 'resnet_v2_101':
            print('variable setting')
            trainable_vars = []
            exclusions = ['resnet_v2_101/logits', 'resnet_v2_101/fully_connected']
            for var in tf.trainable_variables():
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion.strip()):
                        break
                else:
                    trainable_vars.append(var)
            saver_ImageNet = tf.train.Saver(trainable_vars, max_to_keep=10000)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10000)

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            # import pretrained model
            if args.continue_training_dir:
                print('Restoring pretrained model: %s' % args.continue_training_dir)
                saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(args.continue_training_dir)))

            # import pretrained model(ImageNet)
            if args.load_ImageNet == 'True':
                print('Restoring pretrained model(ImageNet): %s' % args.backbone)
                print('./ImageNet(ckpt)/' + args.backbone + '/' + args.backbone +'.ckpt')
                saver_ImageNet.restore(sess, '../ImageNet(ckpt)/' + args.backbone + '/' + args.backbone +'.ckpt')

            # Training loop
            epoch = 0
            min = 100
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                _, loss = train(args, sess, train_set, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
                    batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, global_step,
                    embeddings, total_loss, train_op, args.learning_rate_schedule_file, args.embedding_size)
                # Save variables and the metagraph if it doesn't exist already
                if loss<min and epoch % 5 == 0:
                    min = loss
                    save_variables_and_metagraph(sess, saver, model_dir, subdir, step, epoch)
    return model_dir

def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, global_step,
          embeddings, loss, train_op, learning_rate_schedule_file, embedding_size):
    batch_number = 0
    
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = trinet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)
        # sample_people -> sampling // Input : dataset -> Output : path & number of images in class
        # [ image_path1, image_path2, ... ]
        # [ num_in_class1, num_in_class2, ... ]

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples),(-1,3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))

        for i in range(nrof_batches):
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab,:] = emb

        print('%.3f' % (time.time()-start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
            image_paths, args.people_per_batch, args.alpha)
        # triplets : (anchor, positive, negative) / nrof_random_negs : num_trips / nrof_triplets

        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' % 
            (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        step = 0
        mean_error = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            mean_error += err
            batch_number += 1
            i += 1
            train_time += duration
    mean_error = mean_error/(batch_number+1)
    return step, mean_error
  
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in range(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                # distance cal
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0] # Hard Triplets // VGG Face selecction

                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)

def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person
  
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)
    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1
    return image_paths, num_per_class

def save_variables_and_metagraph(sess, saver, model_dir, model_name, step, epoch):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    model_dir_ = model_dir + '/epoch%d' % epoch
    if not os.path.exists(model_dir_):
        os.makedirs(model_dir_)

    checkpoint_path = os.path.join(model_dir_, 'model-%s-%d.ckpt' % (model_name, epoch))
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)

    metagraph_filename = os.path.join(model_dir_, 'model-%s-%d.meta' % (model_name, epoch))
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # log_base_dir : Directory to logs saved
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='./logs')
    # models_base_dir : Directory to models saved
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='./checkpoints')
    # backbone
    parser.add_argument('--backbone', type=str,
                        choices=['inception_resnet_v2', 'vgg16', 'vgg19',
                                 'resnet_v2_50', 'resnet_v2_101'], default='inception_resnet_v2')
    # load pre-trained ImageNet weights
    parser.add_argument('--load_ImageNet', type=str, default='False')
    # continue_training_dir : directory of pre-train model
    parser.add_argument('--continue_training_dir', type=str,
                        help='Load a pretrained model before training starts.')
    # gpu_memory_fraction : Upper bound of GPU memory
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    # data_dir : Data Directory
    parser.add_argument('--data_dir', type=str, default='../data/cuhk(labeled)/train')
    # # model_def : network model
    # parser.add_argument('--model_def', type=str,
    #     help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v2')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--people_per_batch', type=int,
        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=1e-4)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
