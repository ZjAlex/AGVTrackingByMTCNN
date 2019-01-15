# coding: utf-8
import sys
import os
import numpy as np
import argparse
from toolsFunc.tfrecordUtils import _process_image_withoutcoder, _convert_to_example_simple
import tensorflow as tf

rootPath = "D:/alex/CascadeNetwork3/"


def __iter_all_data(net, iterType):
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    if net not in ['net3']:
        raise Exception("The net type error!")
    if not os.path.isfile(os.path.join(saveFolder, 'pos.txt')):
        raise Exception("Please gen pos.txt in first!")
    if not os.path.isfile(os.path.join(saveFolder, 'landmark.txt')):
        raise Exception("Please gen landmark.txt in first!")
    for line in open(os.path.join(saveFolder, '%s.txt'%(iterType))):
        yield line

def __get_dataset(net, iterType):
    dataset = []
    for line in __iter_all_data(net, iterType):
        info = line.strip().split(' ')
        data_example = dict()
        bbox = dict()
        data_example['filename'] = info[0]
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        bbox['xleft'] = 0
        bbox['yleft'] = 0
        bbox['xright'] = 0
        bbox['yright'] = 0
        if len(info) > 2 and float(info[1]) != -2:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        if len(info) > 2 and float(info[1]) == -2:
            bbox['xleft'] = float(info[2])
            bbox['yleft'] = float(info[3])
            bbox['xright'] = float(info[4])
            bbox['yright'] = float(info[5])

        data_example['bbox'] = bbox
        dataset.append(data_example)
    return dataset

def __add_to_tfrecord(filename, image_example, tfrecord_writer, net):
    """
    Loads data from image and annotations files and add them to a TFRecord.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data, net)
    tfrecord_writer.write(example.SerializeToString())

def gen_tfrecords(net, shuffling=False):
    """
    Runs the conversion operation.
    """
    print(">>>>>> Start tfrecord create...Stage: %s"%(net))
    def _gen(tfFileName, net, iterType, shuffling):
        if tf.gfile.Exists(tfFileName):
            tf.gfile.Remove(tfFileName)
        # GET Dataset, and shuffling.
        dataset = __get_dataset(net=net, iterType=iterType)
        if shuffling:
            np.random.shuffle(dataset)
        # Process dataset files.
        # write the data to tfrecord
        with tf.python_io.TFRecordWriter(tfFileName) as tfrecord_writer:
            for i, image_example in enumerate(dataset):
                if i % 100 == 0:
                    sys.stdout.write('\rConverting[%s]: %d/%d' % (net, i + 1, len(dataset)))
                    sys.stdout.flush()
                filename = image_example['filename']
                __add_to_tfrecord(filename, image_example, tfrecord_writer, net)
        #tfrecord_writer.close()
        print('\n')
    saveFolder = os.path.join(rootPath, "tmp/data/%s/"%(net))
    #tfrecord name
    for n in ['pos', 'landmark']:
        tfFileName = os.path.join(saveFolder, "%s.tfrecord"%(n))
        _gen(tfFileName, net, n, shuffling)
    # Finally, write the labels file:
    print('\nFinished converting the net3 dataset!')
    print('All tf record was saved in %s'%(saveFolder))


if __name__ == "__main__":
    stage = 'net3'
    gen_tfrecords(stage, True)

