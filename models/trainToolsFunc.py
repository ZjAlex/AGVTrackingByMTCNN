import tensorflow as tf
import numpy as np
import cv2


def train_model(baseLr, loss, decay_steps=1000):
    global_step = tf.Variable(0, trainable=False)
    lr_op = tf.train.exponential_decay(baseLr, global_step, decay_steps, 0.1, staircase=True)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


# all mini-batch mirror
def random_flip_images_for_net1_2(image_batch, label_batch):
    # mirror
    if np.random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

    return image_batch


def random_flip_images_for_net3(image_batch, label_batch, landmark_batch):
    # mirror
    if np.random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

            # pay attention: flip landmark
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
            #landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch