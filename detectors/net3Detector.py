import tensorflow as tf
import numpy as np

class Net3Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size * 2, 3], name='input_image')
            #figure out landmark
            self.bbox_pred, self.landmark_pred, self.net = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("Restore param from: ", model_path)
            saver.restore(self.sess, model_path)

        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size

        if len(databatch.shape) <= 3:
            minibatch = [np.array([databatch])]
        else:
            minibatch = []
            cur = 0
            #num of all_data
            n = databatch.shape[0]
            while cur < n:
                #split mini-batch
                minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
                cur += batch_size
        #every batch prediction result
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #bbox_pred batch*4
            bbox_pred,landmark_pred, net = self.sess.run([self.bbox_pred, self.landmark_pred, self.net], feed_dict={self.image_op: data})

            #num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])
            #num_batch * batch_size*10
            landmark_pred_list.append(landmark_pred[:real_size])
            #num_of_data*2,num_of_data*4,num_of_data*10
        return np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
