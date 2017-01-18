import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
from helper import *

def select(imgs, n = 3, fold = 10):
    mags = [None] * (len(imgs)/2)
    for i in range(len(imgs)/2):
        mag, _ = cv2.cartToPolar(np.asarray(imgs[2*i], dtype = np.float32), np.asarray(imgs[2*i+1], dtype = np.float32))
        mags[i] = cv2.norm(mag)
    mags = np.asarray(mags)
    mags /= np.sum(mags)
    idx = []
    tmp = np.random.choice(len(mags), fold*n, replace=False, p = mags)
    idx = np.sort(tmp).reshape(-1, 3)
    return np.asarray(idx)

def save_features(videos, net, transformer, fold):
    """feed rgb flow to caffenet
    and save output of fc7 layer
    """
    n_features = 4096*3
    name = 'siamese'
    for video in videos:
        X = np.empty((0, n_features))
        y = np.empty(0)
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
        flows = load_video(video, './data/ArrowDataAll/', mask = lambda x: x[:3] == 'off', grayscale = True)
        sel_ = select(flows, fold = fold)
        for reverse in [False, True]:
            for flip in [False, True]:
                if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                    direction = 'f'
                    label = 1
                else:
                    direction = 'b'
                    label = 0
                imgs = load_video(video, './data/ArrowDataAll/', mask = lambda x: x[:2] == 'im', grayscale = False, reverse = False, flip = flip)
                if reverse:
                    sel = np.fliplr(sel_)
                else:
                    sel = sel_[:]
                image = []
                for idx in sel:
                    img = np.take(imgs, idx, axis = 0)
                    img = map(lambda x: transformer.preprocess('data', x), img)
                    img = np.asarray(np.concatenate(img, axis = 0))
                    image.append(img)
                image = np.asarray(image)
                net.blobs['triple_data'].data[...] = image
                net.forward()
                X = np.append(X, net.blobs['concat'].data, axis = 0)
                y = np.append(y, np.ones(fold)*label)
        with open(os.path.join('./data/ArrowDataAll', video, 'features-' + name + '.csv'), 'w') as f:
            np.savetxt(f, X, delimiter = ',', fmt = '%f')
        with open(os.path.join('./data/ArrowDataAll', video, 'labels-' + name + '.csv'), 'w') as f:
            np.savetxt(f, y, delimiter = ',', fmt = '%d')
if __name__ == '__main__':
    caffe_root = '/home/bysong/caffe/'
    caffe.set_mode_gpu()

    # load original model
    model_def = './models/caffenet_siamese/deploy.prototxt'
    model_weights = './models/caffenet_siamese/caffenet_train_iter_100000.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.ones(3)*np.mean(np.load('./data/mean.npy')))

    train_list, test_list = load_list('./data', dataset = 1)
    videos = train_list
    videos.extend(test_list)


    fold = 10
    net.blobs['triple_data'].reshape(fold, 9, 227, 227)

    save_features(videos, net, transformer, fold)
