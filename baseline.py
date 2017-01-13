import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
from helper import *

def save_features(videos, net, transformer):
    """feed rgb flow to caffenet
    and save output of fc7 layer
    """
    n_features = 4096
    name = 'baseline'
    for video in videos:
        X = np.empty((0, n_features))
        y = np.empty(0)
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
        for reverse in [False, True]:
            for flip in [False, True]:
                if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                    direction = 'f'
                else:
                    direction = 'b'
                flows = load_video(video, './data/ArrowDataAll/', mask = lambda x: x[:3] == 'of' + direction, grayscale = True, flip = flip)
                sel = np.asarray([[2*i, 2*i+1] for i in select(flows, 1)]).flatten()
                flows = np.take(flows, sel, axis = 0)
                imgs = []
                for i in range(len(flows)/2):
                    _, ang = cv2.cartToPolar(np.asarray(flows[0], dtype = np.float32), np.asarray(flows[1], dtype = np.float32))
                    image = np.stack([flows[0], flows[1], cv2.normalize(ang,None,0,255,cv2.NORM_MINMAX)], axis = -1)
                    imgs.append(image)
                imgs = map(lambda x: transformer.preprocess('data', x), imgs)
                net.blobs['data'].data[...] = imgs
                net.forward()
                X = np.append(X, net.blobs['fc7'].data, axis = 0)
                if direction == 'f':
                    y = np.append(y, 1)
                else:
                    y = np.append(y, 0)
        with open(os.path.join('./data/ArrowDataAll', video, 'features-' + name + '.csv'), 'w') as f:
            np.savetxt(f, X, delimiter = ',', fmt = '%f')
        with open(os.path.join('./data/ArrowDataAll', video, 'labels-' + name + '.csv'), 'w') as f:
            np.savetxt(f, y, delimiter = ',', fmt = '%d')

if __name__ == '__main__':
    caffe_root = '/home/bysong/caffe/'
    caffe.set_mode_gpu()
    # load original model
    model_def = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
    model_weights = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    net = caffe.Net(model_def, model_weights, caffe.TEST) 
    net.blobs['data'].reshape(1, 3, 227, 227)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', np.asarray([128, 128, 128]))            # subtract the dataset-mean value in each channel

    train_list, test_list = load_list('./data', dataset = 1)
    videos = train_list
    videos.extend(test_list)
    save_features(videos, net, transformer)
