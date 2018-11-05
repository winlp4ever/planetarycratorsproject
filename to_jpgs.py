from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    data_train = np.load('./data/data_train.npy')
    data_test = np.load('./data/data_test.npy')

    train_dir = './data/train'
    test_dir ='./data/test'

    train = (data_train, train_dir)
    test = (data_test, test_dir)

    for mode in (train, test):
        data = mode[0]
        dir = mode[1]
        for i in range(data.shape[0]):
            im_size = data[i].shape
            print('{}: {}'.format(i, im_size))
            im = np.empty(im_size + (3,))
            im[:, :, 0] = im[:, :, 1] = im[:, :, 2] = data_train[i]
            img = Image.fromarray(im.astype('uint8'))
            img.save(os.path.join(dir, '{}.jpg'.format(i)))