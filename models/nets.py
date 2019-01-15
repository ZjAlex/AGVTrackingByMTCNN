
from tensorflow.contrib import slim
from tensorflow.python.keras import layers
from models.lossFunc import *

# cls and box regression
def Net_1(inputs, label=None, bbox_target=None, training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.separable_conv2d(inputs, 10, 3, 1, stride=1, scope='conv1', padding='valid')
        # net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1')
        net = slim.separable_conv2d(net, 16, kernel_size=[3, 6], depth_multiplier=1, stride=1, scope='conv2', padding='valid')
        #net = slim.conv2d(net, num_outputs=16, kernel_size=[3,6], stride=1, scope='conv2')
        net = slim.separable_conv2d(net, 32, kernel_size=[3, 6], depth_multiplier=1, stride=1, scope='conv3', padding='valid')
       # net = slim.conv2d(net, num_outputs=32, kernel_size=[3,6], stride=1, scope='conv3')
        #batch*H*W*2
        conv4_1 = slim.separable_conv2d(net, 2, kernel_size=[1, 1], depth_multiplier=1, stride=1, scope='conv4_1', activation_fn=tf.nn.softmax, padding='valid')
        #conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1,1], stride=1, scope='conv4_1', activation_fn=tf.nn.softmax)
        #batch*H*W*4
        bbox_pred = slim.separable_conv2d(net, 4, kernel_size=[1, 1], depth_multiplier=1, stride=1, scope='conv4_2',
                                        activation_fn=None, padding='valid')
        #bbox_pred = slim.conv2d(net, num_outputs=4, kernel_size=[1,1], stride=1, scope='conv4_2', activation_fn=None)
        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1, [1,2], name='cls_prob')
            cls_loss = cls_ohem(cls_prob, label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)

            accuracy = cal_accuracy(cls_prob, label)
           # L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, accuracy
        else: # testing
            #when test, batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
            return cls_pro_test, bbox_pred_test


# cls and box regression
def Net_2(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.separable_conv2d(inputs, 28, [3, 6], 1, scope='conv1', padding='valid')
        # net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,6], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        net = slim.separable_conv2d(net, 48, [3, 6], 1, scope='conv2', padding='valid')
        #net = slim.conv2d(net,num_outputs=48,kernel_size=[3,6],stride=[1, 1],scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        net = slim.separable_conv2d(net, 64, [2, 4], 1, scope='conv3', padding='valid')
      #  net = slim.conv2d(net,num_outputs=64,kernel_size=[2,4],stride=1,scope="conv3")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
#            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, accuracy
        else:
            return cls_prob, bbox_pred


def Net_2_v2(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,6], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        net = slim.conv2d(net,num_outputs=48,kernel_size=[3,6],stride=[1, 1],scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[2,4],stride=1,scope="conv3")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1")
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)

        landmark_pred = slim.fully_connected(fc1, num_outputs=4, scope="landmark_fc", activation_fn=None)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss,L2_loss, accuracy
        else:
            return cls_prob, bbox_pred, landmark_pred


# box regression and landmark regression
def Net_3(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.separable_conv2d(inputs, 32, [3, 6], 1, scope='conv1', padding='valid')
        #net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,6], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        net = slim.separable_conv2d(net, 64, [3, 6], 1, scope='conv2', padding='valid')
        #net = slim.conv2d(net,num_outputs=64,kernel_size=[3,6],stride=1,scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool2")
        net = slim.separable_conv2d(net, 64, [3, 6], 1, scope='conv3', padding='valid')
        #net = slim.conv2d(net,num_outputs=64,kernel_size=[3,6],stride=1,scope="conv3")
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool3")
        net = slim.separable_conv2d(net, 128, [2, 4], 1, scope='conv4', padding='valid')
       # net = slim.conv2d(net,num_outputs=128,kernel_size=[2,4],stride=1,scope="conv4")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #batch*4
        landmark_pred = slim.fully_connected(fc1,num_outputs=4,scope="landmark_fc",activation_fn=None)
        #train
        if training:
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
          #  L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return bbox_loss,landmark_loss
        else:
            return bbox_pred, landmark_pred, net



def Net_3_V2(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        net = slim.max_pool2d(net, kernel_size=[2,2],stride=2,scope="pool1")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool2")
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        net = slim.max_pool2d(net,kernel_size=[2,2],stride=2,scope="pool3")
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        fc_flatten = slim.flatten(net)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1")
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        #batch*4
        landmark_pred = slim.fully_connected(fc1,num_outputs=4,scope="landmark_fc",activation_fn=None)
        #train
        if training:
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            landmark_loss = landmark_ohem_v2(landmark_pred, landmark_target,label)
            L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            return bbox_loss,landmark_loss,L2_loss
        else:
            return bbox_pred, landmark_pred
