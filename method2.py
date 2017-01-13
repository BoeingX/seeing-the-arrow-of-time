import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os

def load_list(data_dir, prefix = None, dataset = 1):
    train_list = os.path.join(data_dir, 'train') + str(dataset) + '.idx'
    test_list = os.path.join(data_dir, 'test') + str(dataset) + '.idx'
    with open(train_list) as f:
        train_list = f.read().splitlines()
    with open(test_list) as f:
        test_list = f.read().splitlines()
    if prefix is not None:
        train_list = map(lambda x: os.path.join(prefix, x), train_list)
        test_list = map(lambda x: os.path.join(prefix, x), test_list)
    return train_list, test_list

def load_img(filename, flip = False, grayscale = False):
    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    if grayscale:
        width, height = img.shape
    else:
        width, height, _ = img.shape
    factor = max(256.0 / width, 256.0 / height)
    img = cv2.resize(img, None, fx = factor, fy = factor)
    return img

def load_imgs(filenames, reverse = False, flip = False, grayscale = False):
    if reverse:
        filenames = filenames[::-1]
    imgs = map(lambda x: load_img(x, flip, grayscale), filenames)
    return imgs

def load_video(video, data_dir, mask = None, reverse = False, flip = False, grayscale = False):
    filenames = os.listdir(os.path.join(data_dir, video))
    filenames = filter(lambda x: x[-4:] == 'jpeg', filenames)
    if mask is not None:
        filenames = filter(mask, filenames)
    filenames.sort()
    filenames = map(lambda x: os.path.join(data_dir, video, x), filenames)
    imgs = load_imgs(filenames, reverse, flip, grayscale)
    return imgs

def is_forward(video):
    if video[0] == 'F':
        return True
    return False



def load_flows(video):
    images = []
    labels = []
    for direction in ['f', 'b']:
        for flip in [False, True]:
            imgs_ = load_video(video, './data/ArrowDataAll', 
                               mask = lambda x: x[:3] == 'of' + direction, 
                               flip = flip, 
                               grayscale=True)
            imgs_ = map(lambda x: cv2.resize(x, (227, 227)), imgs_)
            imgs = []
            for i in range(len(imgs_)/2):
                imgs.append(np.asarray([imgs_[2*i], imgs_[2*i+1]]))
            images.extend(imgs)
            if direction == 'f':
                labels.extend([1]*len(imgs))
            else:
                labels.extend([0]*len(imgs))
    images = np.asarray(images)
    return images, np.asarray(labels)

def find_optimal_batch(n):
    n_opt = 4
    i = 4
    while i <= min(n, 20):
        if n % i == 0:
            n_opt = i
        i += 4
    return n_opt

def forward(videos, net):
    X = np.empty((0, 4096))
    y = np.empty(0)
    video_names = []
    for video in videos:
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
        images, labels = load_flows(video)
        images -= 128*np.ones_like(images)
        n_opt = find_optimal_batch(len(images))
        print '[INFO] %d images' % len(images)
        print '[INFO] processing batch of %d' % n_opt
        net.blobs['data'].reshape(n_opt, 2, 227, 227)
        for i in range(len(images)/n_opt):
            net.blobs['data'].data[...] = images[n_opt*i:n_opt*(i+1),...]
            output = net.forward()
            X = np.append(X, output['fc7'], axis = 0)
        y = np.append(y, labels)
        video_names.extend([video] * len(labels))
    return X, y, video_names

def forward_save(videos, net):
    for video in videos:
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
        X = np.empty((0, 4096))
        y = np.empty(0)
        images, labels = load_flows(video)
        images -= 128*np.ones_like(images)
        n_opt = find_optimal_batch(len(images))
        print '[INFO] %d images' % len(images)
        print '[INFO] processing batch of %d' % n_opt
        net.blobs['data'].reshape(n_opt, 2, 227, 227)
        for i in range(len(images)/n_opt):
            net.blobs['data'].data[...] = images[n_opt*i:n_opt*(i+1),...]
            output = net.forward()
            X = np.append(X, output['fc7'], axis = 0)
        y = np.append(y, labels)
        with open(os.path.join('./data/ArrowDataAll', video, 'features.csv'), 'w') as f:
            np.savetxt(f, X, delimiter = ',', fmt = '%f')
        with open(os.path.join('./data/ArrowDataAll', video, 'labels.csv'), 'w') as f:
            np.savetxt(f, y, delimiter = ',', fmt = '%d')

if __name__ == '__main__':
    caffe.set_mode_gpu()
    model_def = 'models/bvlc_reference_caffenet/deploy.prototxt'
    model_weights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 
    conv1_mean = net.params['conv1'][0].data.mean(axis = 1)
    conv1_stack = np.repeat(conv1_mean[:, np.newaxis, :, :], 2, axis = 1)
    model_def = 'models/caffenet2/deploy.prototxt'
    model_weights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 
    net.params['conv1-stack'][0].data[...] = conv1_stack

    train_list, test_list = load_list('./data', dataset = 1)
    #X_train, y_train, train_list_full = forward(train_list, net)
    #X_train, y_test, test_list_full = forward(test_list, net)

    #np.savetxt('X_train1.csv', X_train, delimiter = ',')
    #np.savetxt('y_train1.csv', y_train, delimiter = ',')
    #np.savetxt('X_test1.csv', X_test, delimiter = ',')
    #np.savetxt('y_test1.csv', y_test, delimiter = ',')
    #with open('videos_train1.csv', 'w') as f:
    #    map(lambda x: f.write(str(x) + '\n'), train_list_full)
    #with open('videos_test1.csv', 'w') as f:
    #    map(lambda x: f.write(str(x) + '\n'), test_list_full)

    videos = train_list
    videos.extend(test_list)
    forward_save(videos, net)
