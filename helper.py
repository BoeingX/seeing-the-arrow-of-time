import os
import sys
import cv2
import numpy as np
def show_imgs(imgs):
    for img in imgs:
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, (256, 256))
    return img

def load_imgs(filenames, reverse = False, flip = False):
    if reverse:
        filenames = filenames[::-1]
    imgs = map(lambda x: load_img(x, flip), filenames)
    return imgs

def load_video(video, data_dir, mask = None, reverse = False, flip = False):
    filenames = os.listdir(os.path.join(data_dir, video))
    filenames = filter(lambda x: x[-4:] == 'jpeg', filenames)
    if mask is not None:
        filenames = filter(mask, filenames)
    filenames.sort()
    filenames = map(lambda x: os.path.join(data_dir, video, x), filenames)
    imgs = load_imgs(filenames, reverse, flip)
    return imgs

def is_forward(video):
    if video[0] == 'F':
        return True
    return False

def load_features(video, data_dir, suffix):
    with open(os.path.join(data_dir, video, 'features' + '_' + suffix + '.csv'), 'r') as f:
        X = np.loadtxt(f, delimiter = ',')
    return X

def load_labels(video):
    if is_forward(video):
        y = np.asarray([1, 1, -1, -1])
    else:
        y = np.asarray([-1, -1, 1, 1])
    return y

def load_features_labels(videos, data_dir, suffix):
    X = np.asarray(map(lambda x: load_features(x, data_dir, suffix), videos))
    X = X.reshape(-1, X.shape[-1])
    y = np.asarray(map(load_labels, videos))
    y = y.reshape(1, -1).squeeze()
    return X, y

def load_dataset(train_list, test_list, data_dir, suffix):
    X_train, y_train = load_features_labels(train_list, data_dir, suffix)
    X_test, y_test = load_features_labels(test_list, data_dir, suffix)
    return X_train, y_train, X_test, y_test

def load_list(data_dir, dataset = 1, prefix = True):
    train_list = os.path.join(data_dir, 'train') + str(dataset) + '.idx'
    test_list = os.path.join(data_dir, 'test') + str(dataset) + '.idx'
    with open(train_list) as f:
        train_list = f.read().splitlines()
    with open(test_list) as f:
        test_list = f.read().splitlines()
    if prefix:
        train_list = map(lambda x: os.path.join(data_dir, x), train_list)
        test_list = map(lambda x: os.path.join(data_dir, x), test_list)
    return train_list, test_list

def is_raw_img(filename):
    return filename[:2] == 'im'
