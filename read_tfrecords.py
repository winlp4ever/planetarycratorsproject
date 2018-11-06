import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.patches as patches


data_path = './data/train.tfrecord'


def decode(serialized_example):
    ex = tf.parse_single_example(
        serialized_example,
        features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        })
    id = ex['image/source_id']
    im_ts = tf.image.decode_jpeg(ex['image/encoded'], channels=3)
    xmins = tf.sparse_tensor_to_dense(ex['image/object/bbox/xmin'], default_value=0)
    xmaxs = tf.sparse_tensor_to_dense(ex['image/object/bbox/xmax'], default_value=0)
    ymins = tf.sparse_tensor_to_dense(ex['image/object/bbox/ymin'], default_value=0)
    ymaxs = tf.sparse_tensor_to_dense(ex['image/object/bbox/ymax'], default_value=0)

    return im_ts, id, xmins, xmaxs, ymins, ymaxs

def parse_and_demo(session, ex_tensor):
    im, idx, xmins, xmaxs, ymins, ymaxs = session.run(ex_tensor)
    print(idx.decode('utf8'))
    #img = Image.fromarray(np.reshape(im, [224, 224, 3]))
    #img.show(str(''))
    xmins, xmaxs, ymins, ymaxs = map(lambda arr: arr * 224, (xmins, xmaxs, ymins, ymaxs))
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    nb = 0
    if len(xmins) > 0:
        for i in range(len(xmins)):
            cter = ((xmins[i] + xmaxs[i]) / 2, (ymins[i] + ymaxs[i]) / 2)
            r = max((xmaxs[i] - xmins[i]) / 2, (ymaxs[i] - ymins[i]) / 2)
            if cter[0] + r > 224 or cter[1] + r > 224:
                nb += 1
            crater = patches.Circle(cter, r, color='r', alpha=0.5)
            ax.add_patch(crater)
            print('center {} radius {}'.format(cter, r))
    #print(zip(xmins, ymins))
    if nb > 0:
        fig.show()
        plt.pause(5)
        plt.close()
    else:
        plt.close()

with tf.Session() as sess:
    dataset = tf.data.TFRecordDataset([data_path]).map(decode)
    iterator = dataset.make_one_shot_iterator()
    ex = iterator.get_next()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    i = 0
    try:
        while True:
            i += 1
            parse_and_demo(sess, ex_tensor=ex)
    except tf.errors.OutOfRangeError:
        # Raised when we reach the end of the file.
        pass
    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()
