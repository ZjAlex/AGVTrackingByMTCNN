# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
import argparse
from toolsFunc.tfrecordReaderUtils import read_single_tfrecord_for_net1_2, read_multi_tfrecords_v2
from models.configs import config
from models.nets import Net_1, Net_2, Net_3,Net_2_v2
import cv2
from six import string_types, iteritems
from models.trainToolsFunc import train_model, random_flip_images_for_net1_2, random_flip_images_for_net3
rootPath = "D:/alex/CascadeNetwork3/"


def train(netFactory, modelPrefix, endEpoch, dataPath, display=200, baseLr=0.01, gpus=""):
    net = modelPrefix.split('/')[-1]
    print("Now start to train...stage: %s" % (net))
    # set GPU
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus


    pos_dir = os.path.join(dataPath, 'pos.tfrecord')
    part_dir = os.path.join(dataPath, 'part.tfrecord')
    neg_dir = os.path.join(dataPath, 'neg.tfrecord')
    landmark_dir = os.path.join(dataPath, 'landmark.tfrecord')
    dataset_dirs = [pos_dir, part_dir, neg_dir, landmark_dir]
    pos_ratio, part_ratio, landmark_ratio, neg_ratio = 1 / 6, 2 / 6, 2 / 6, 1 / 6
    pos_batch_size = int(np.ceil(config.BATCH_SIZE * pos_ratio))
    part_batch_size = int(np.ceil(config.BATCH_SIZE * part_ratio))
    neg_batch_size = int(np.ceil(config.BATCH_SIZE * neg_ratio))
    landmark_batch_size = int(np.ceil(config.BATCH_SIZE * landmark_ratio))
    batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]
    image_batch, label_batch, bbox_batch, landmark_batch = read_multi_tfrecords_v2(dataset_dirs, batch_sizes, net)
    total_num = 0
    for d in dataset_dirs:
        total_num += sum(1 for _ in tf.python_io.tf_record_iterator(d))
    # ratio
    if net == 'pnet':
        image_size = 12
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 0.5, 0.1
    elif net == 'net2_v2':
        image_size = 24
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 1.0, 1.0, 1.0
    elif net == 'onet':
        ratio_cls_loss, ratio_bbox_loss, ratio_landmark_loss = 0.0, 1.0, 1.0
        image_size = 48
    else:
        raise Exception("incorrect net type.")
    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size * 2, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='landmark_target')
    # class,regression
    cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, accuracy_op = netFactory(input_image, label, bbox_target,
                                                                                      landmark_target, training=True)

    # train,update learning rate(3 loss)
    train_op, lr_op = train_model(baseLr,
                                  ratio_cls_loss * cls_loss_op + ratio_bbox_loss * bbox_loss_op + ratio_landmark_loss * landmark_loss_op + L2_loss_op,
                                  10000)
    # init
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    data_path = 'D:/alex/facenet-master/src/align/det2.npy'
    data_dict = np.load(data_path, encoding='latin1').item()
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for param_name, data in iteritems(data_dict[op_name]):
                try:
                    var = tf.get_variable(param_name)
                    sess.run(var.assign(data))
                    print(param_name)
                    print('\n')
                except ValueError:
                    if not True:
                        raise



    # save model
    saver = tf.train.Saver(max_to_keep=0)
   # model_path = 'D:/alex/tmp/model/onet_2'
   # ckpt = tf.train.get_checkpoint_state(model_path)
   # saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)
    # visualize some variables
    tf.summary.scalar("cls_loss", cls_loss_op)  # cls_loss
    tf.summary.scalar("bbox_loss", bbox_loss_op)  # bbox_loss
    tf.summary.scalar("landmark_loss", landmark_loss_op)  # landmark_loss
    tf.summary.scalar("cls_accuracy", accuracy_op)  # cls_acc
    summary_op = tf.summary.merge_all()
    logs_dir = os.path.join(rootPath, "tmp", "logs", net)
    if os.path.exists(logs_dir) == False:
        os.makedirs(logs_dir)
    writer = tf.summary.FileWriter(logs_dir, sess.graph)
    # begin
    coord = tf.train.Coordinator()
    # begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    # total steps
    MAX_STEP = int(total_num / config.BATCH_SIZE + 1) * endEpoch
    print("\n\nTotal step: ", MAX_STEP)
    epoch = 0
    sess.graph.finalize()
    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(
                [image_batch, label_batch, bbox_batch, landmark_batch])
            # random flip
            image_batch_array, landmark_batch_array = random_flip_images_for_net3(image_batch_array, label_batch_array,
                                                                         landmark_batch_array)
            '''
            print image_batch_array.shape
            print label_batch_array.shape
            print bbox_batch_array.shape
            print landmark_batch_array.shape
            print label_batch_array[0]
            print bbox_batch_array[0]
            print landmark_batch_array[0]
            '''
            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array, label: label_batch_array,
                                                bbox_target: bbox_batch_array, landmark_target: landmark_batch_array})

            if (step + 1) % display == 0:
                # acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss, landmark_loss, L2_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, landmark_loss_op, L2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,
                               landmark_target: landmark_batch_array})
                print(
                    "%s [%s] Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f, landmark loss: %4f,L2 loss: %4f,lr:%f " % (
                        datetime.now(), net, step + 1, acc, cls_loss, bbox_loss, landmark_loss, L2_loss, lr))
            # save every two epochs
            if i * config.BATCH_SIZE > total_num * 2:
                epoch = epoch + 1
                i = 0
                saver.save(sess, modelPrefix, global_step=epoch * 2)
            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        print("Done!")
    finally:
        coord.request_stop()
        writer.close()
    coord.join(threads)
    sess.close()


if __name__ == "__main__":
    dataPath = os.path.join(rootPath, "tmp/data/%s" % ("net2_v2"))
    modelPrefix = os.path.join(rootPath, "tmp/model/%s/%s" % ("net2_v2", "net2_v2"))
    if not os.path.isdir(os.path.dirname(modelPrefix)):
        os.makedirs(os.path.dirname(modelPrefix))
    display_steps = 100
    epoch = 120
    lr = 0.01

    train(Net_2_v2, modelPrefix, epoch, dataPath, display=display_steps, baseLr=lr, gpus='0')
