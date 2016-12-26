#!/usr/bin/env python2
import os
import cv2
import numpy as np
import caffe
import progressbar

def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    #fy = 983.0 / img.shape[1]
    #img = cv2.resize(img, (0, 0), fx = fy, fy = fy)
    img = cv2.resize(img, (227, 227))
    return img

def load_imgs(filenames, reverse = False, flip = False):
    if reverse:
        filenames = filenames[::-1]
    imgs = map(lambda x: load_img(x, flip), filenames)
    return imgs

def load_video(filename, reverse = False, flip = False):
    filenames = os.listdir(filename)
    filenames = filter(lambda x: x[-4:] == 'jpeg', filenames)
    filenames.sort()
    filenames = map(lambda x: os.path.join(filename, x), filenames)
    imgs = load_imgs(filenames, reverse, flip)
    return imgs

def optical_flow(imgs):
    T = len(imgs)
    flows = []
    for i in range(1, T-1):
        flow = cv2.calcOpticalFlowFarneback(imgs[i-1],imgs[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
    return flows

def load_data(dataset = 1):
    index_dir = './helper'
    train_list = os.path.join(index_dir, 'train') + str(dataset) + '.txt'
    test_list = os.path.join(index_dir, 'test') + str(dataset) + '.txt'
    with open(train_list) as f:
        train_list = f.read().splitlines()
    with open(test_list) as f:
        test_list = f.read().splitlines()
    train_labels = map(lambda x: x[0] == 'F', train_list)
    test_labels = map(lambda x: x[0] == 'F', test_list)
    return train_list, train_labels, test_list, test_labels

def optical_flow_to_motion(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def motion_to_vec(motions, net):
    net.blobs['data'].reshape(len(motions), 3, 227, 227)
    net.blobs['data'].data[...] = motions
    ### perform classification
    output = net.forward()
    output_prob = output['prob']
    return np.mean(output_prob, axis = 0)

def video_to_vec(video, net, transformer):
    predictions = [None] * 4
    idx = 0
    for reverse in [False, True]:
        for flip in [False, True]:
            imgs = load_video(video, reverse, flip)
            imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
            flows = optical_flow(imgs_gray)
            motions = map(optical_flow_to_motion, flows)
            motions = map(lambda x: transformer.preprocess('data', x), motions)
            predictions[idx] = motion_to_vec(motions, net)
            idx += 1
    return np.asarray(predictions)

def write_features(predictions, video):
    with open(os.path.join(video, 'features.csv'), 'w') as f:
        np.savetxt(f, predictions, delimiter = ',')

def run():
    CAFFE_ROOT = '/home/bysong/caffe/'
    MODEL_FILE = '/home/bysong/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    PRETRAINED = '/home/bysong/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST) 
    mu = np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    #net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    #                   mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
    #                   channel_swap=(2,1,0),
    #                   raw_scale=255,
    #                   image_dims=(256, 256))

    data_dir = "./data/ArrowDataAll/"
    videos = map(lambda x: os.path.join(data_dir, x), os.listdir(data_dir))

    count = 1
    for video in videos:
        print '[INFO] processing video %d / %d' % (count, len(videos))
        predictions = video_to_vec(video, net, transformer)

        write_features(predictions, video)
        count += 1

if __name__ == '__main__':
    run()
    #video = "./data/ArrowDataAll/F_aqvxyejK0MQ/"
    #imgs = load_video(video, False, False)
    #imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
    #flows = optical_flow(imgs_gray)
    #motions = map(optical_flow_to_motion, flows)
    #caffe_root = '/home/bysong/caffe/'  # this file is expected to be in {caffe_root}/examples
    #MODEL_FILE = '/home/bysong/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    ##MODEL_FILE = 'deploy.prototxt'
    #PRETRAINED = '/home/bysong/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    #caffe.set_mode_gpu()
    #net = caffe.Classifier(MODEL_FILE, PRETRAINED,
    #                   mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
    #                   channel_swap=(2,1,0),
    #                   raw_scale=255,
    #                   image_dims=(256, 256))
    #predictions = video_to_vec(video, net)
    #with open(video + 'features.csv', 'w') as f:
    #    np.savetxt(f, predictions, delimiter = ',')

    #for flow in flows:
    #    cv2.imshow('frame', optical_flow_to_img(flow))
    #    k = cv2.waitKey(30) & 0xff
    #    if k == 27:
    #        break
    #cv2.destroyAllWindows()
