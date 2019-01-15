# coding:utf-8
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import sys
import argparse
from toolsFunc.tfrecordReaderUtils import read_single_tfrecord_for_net1_2
from models.trainToolsFunc import train_model, random_flip_images_for_net1_2
from models.nets import Net_1
import cv2
from six import string_types, iteritems
from models.configs import config
rootPath = "D:/alex/CascadeNetwork3/"


def train(netFactory, modelPrefix, endEpoch, dataPath, display=100, baseLr=0.01, gpus=""):
    print("Now start to train stage: net1")
    net = 'net1'
    # set GPU
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    dataset_dir = os.path.join(dataPath, 'all.tfrecord')
    total_num = sum(1 for _ in tf.python_io.tf_record_iterator(dataset_dir))
    image_batch, label_batch, bbox_batch = read_single_tfrecord_for_net1_2(dataset_dir, config.BATCH_SIZE, net)

    # ratio
    image_size = 12
    ratio_cls_loss, ratio_bbox_loss = 1.0, 1.0
    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size * 2, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    # class,regression
    cls_loss_op, bbox_loss_op, accuracy_op = netFactory(input_image, label, bbox_target, training=True)

    # train,update learning rate(3 loss)
    train_op, lr_op = train_model(baseLr,
                                  ratio_cls_loss * cls_loss_op + ratio_bbox_loss * bbox_loss_op,
                                  10000)
    # init
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # 加载预训练模型权重
    data_path = 'D:/alex/facenet-master/src/align/det1.npy'
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
    tf.summary.scalar("bbox_loss", bbox_loss_op)  # bbox_los
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
            image_batch_array, label_batch_array, bbox_batch_array = sess.run(
                [image_batch, label_batch, bbox_batch])
            # random flip
            image_batch_array = random_flip_images_for_net1_2(image_batch_array, label_batch_array)
            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array, label: label_batch_array,
                                                bbox_target: bbox_batch_array})

            if (step + 1) % display == 0:
                # acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array})
                print(
                    "%s [%s] Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f " % (
                        datetime.now(), net, step + 1, acc, cls_loss, bbox_loss, lr))
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
    dataPath = os.path.join(rootPath, "tmp/data/%s" % ("net1"))
    modelPrefix = os.path.join(rootPath, "tmp/model/%s/%s" % ("net1", "net1"))
    if not os.path.isdir(os.path.dirname(modelPrefix)):
        os.makedirs(os.path.dirname(modelPrefix))
    display_steps = 100
    epoch = 100
    lr = 0.01
    train(Net_1, modelPrefix, epoch, dataPath, display=display_steps, baseLr=lr, gpus='0')

