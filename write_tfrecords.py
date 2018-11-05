import tensorflow as tf
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from PIL import Image
import io
import sys
import os
import glob
#sys.path.append("/media/redlcamille/DATA/tensorflow/models/research/object_detection")
#from object_detection.utils import dataset_util

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _to_tf_example(impath, label):
    with tf.gfile.GFile(impath, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    width, height = image.size

    fn = os.path.basename(impath)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    class_texts = []
    if label.size != 0:
        for y, x, radius in label:
            xmins.append(max(x - radius, 0.) / width)
            xmaxs.append(min(x + radius, width) / width)
            ymins.append(max(y - radius, 0.) / height)
            ymaxs.append(min(y + radius, height) / height)
            classes.append(1)
            class_texts.append('marsCraters'.encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/filename': _bytes_feature(str(fn).encode('utf8')),
        'image/source_id': _bytes_feature(str(fn).encode('utf8')),
        'image/encoded': _bytes_feature(encoded_jpg),
        'image/format': _bytes_feature(b'jpg'),
        'image/object/bbox/xmin': _float_list_feature(xmins),
        'image/object/bbox/xmax': _float_list_feature(xmaxs),
        'image/object/bbox/ymin': _float_list_feature(ymins),
        'image/object/bbox/ymax': _float_list_feature(ymaxs),
        'image/object/class/text': _bytes_list_feature(class_texts),
        'image/object/class/label': _int64_list_feature(classes),
    }))
    return example

if __name__ == '__main__':
    train_filename = './data/train.tfrecord'
    writer = tf.python_io.TFRecordWriter(train_filename)

    train_dir = './data/train'
    train_label = './data/labels_train.csv'

    y_train = pd.read_csv(train_label)

    for idx in range(9000):
        im_path = os.path.join(train_dir, '{}.jpg'.format(idx))
        label = y_train[y_train['i'] == idx][['row_p', 'col_p', 'radius_p']].values

        tf_example = _to_tf_example(im_path, label)
        print(idx)
        writer.write(tf_example.SerializeToString())

    writer.close()
    sys.stdout.flush()
