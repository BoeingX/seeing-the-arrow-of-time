#!/usr/bin/env python2
import os
import cv2
import numpy as np
from helper import *
from sklearn.svm import SVC

# first option
# generate a feature for each optical flow
# aggregate them as the descriptor feeding to SVM
def motion_to_vec(motion, net):
    net.blobs['data'].data[...] = motion
    output = net.forward()
    output_prob = output['prob']
    return output_prob

def motions_to_vec(motions, net):
    output_prob = np.asarray(map(lambda x: motion_to_vec(x, net), motions))
    return np.mean(output_prob, axis = 0).squeeze()

def video_to_vec(video, data_dir, net, transformer):
    predictions = [None] * 4
    idx = 0
    for reverse in [False, True]:
        for flip in [False, True]:
            imgs = load_video(video, data_dir, is_raw_image, reverse, flip)
            imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
            flows = optical_flow(imgs_gray)
            motions = map(optical_flow_to_motion, flows)
            motions = map(lambda x: transformer.preprocess('data', x), motions)
            predictions[idx] = motions_to_vec(motions, net)
            idx += 1
    return np.asarray(predictions)

def write_features(predictions, data_dir, suffix, video):
    with open(os.path.join(data_dir, video, 'features' + '_' + suffix + '.csv'), 'w') as f:
        np.savetxt(f, predictions, delimiter = ',')

def generate_features():
    data_dir = 'data/ArrowDataAll'
    suffix = 'bvlc_reference_caffenet'
    caffe_root = os.path.join(os.path.expanduser('~'), 'caffe')
    model_file = os.path.join(caffe_root, suffix, 'deploy.prototxt')
    pretrained = os.path.join(caffe_root, suffix + '.caffemodel')

    caffe.set_mode_gpu()
    net = caffe.Net(model_file, pretrained, caffe.TEST) 
    net.blobs['data'].reshape(1, 3, 227, 227)
    mu = np.load(os.path.join(CAFFE_ROOT, 'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    mu = mu.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    videos = os.listdir(data_dir)
    count = 1
    for video in videos:
        print '[INFO] processing video %d / %d' % (count, len(videos))
        predictions = video_to_vec(video, data_dir, net, transformer)
        write_features(predictions, data_dir, video)
        count += 1

# second option
# save optical flows to disk
# then fine tune a 2-class CNN

if __name__ == '__main__':
    pass
