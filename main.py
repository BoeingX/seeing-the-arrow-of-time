#!/usr/bin/env python2
import os
import cv2
import numpy as np
import caffe
import progressbar
from sklearn.svm import SVC

INDEX_DIR = 'helper'
DATA_DIR = 'data/ArrowDataAll'
SUFFIX = 'bvlc_reference_caffenet'
CAFFE_ROOT = os.path.join(os.path.expanduser('~'), 'caffe')
MODEL_FILE = os.path.join(CAFFE_ROOT, SUFFIX, 'deploy.prototxt')
PRETRAINED = os.path.join(CAFFE_ROOT, SUFFIX, SUFFIX + '.caffemodel')

def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    #fy = 983.0 / img.shape[1]
    #img = cv2.resize(img, (0, 0), fx = fy, fy = fy)
    img = cv2.resize(img, (256, 256))
    return img

def load_imgs(filenames, reverse = False, flip = False):
    if reverse:
        filenames = filenames[::-1]
    imgs = map(lambda x: load_img(x, flip), filenames)
    return imgs

def load_video(video, reverse = False, flip = False):
    filenames = os.listdir(video)
    filenames = filter(lambda x: x[:2] == 'im' and x[-4:] == 'jpeg', filenames)
    filenames.sort()
    filenames = map(lambda x: os.path.join(video, x), filenames)
    imgs = load_imgs(filenames, reverse, flip)
    return imgs

def optical_flow(imgs):
    T = len(imgs)
    flows = []
    for i in range(1, T-1):
        flow = cv2.calcOpticalFlowFarneback(imgs[i-1],imgs[i+1], None, 0.5, 3, 15, 3, 5, 1.1, 0)
        flows.append(flow)
    return flows

def optical_flow_to_motion(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def motion_to_vec(motion, net):
    net.blobs['data'].data[...] = motion
    output = net.forward()
    output_prob = output['prob']
    return output_prob

def motions_to_vec(motions, net):
    output_prob = np.asarray(map(lambda x: motion_to_vec(x, net), motions))
    return np.mean(output_prob, axis = 0).squeeze()

def video_to_vec(video, net, transformer):
    predictions = [None] * 4
    idx = 0
    for reverse in [False, True]:
        for flip in [False, True]:
            imgs = load_video(video, reverse, flip)
            imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
            flows = optical_flow(imgs_gray)
            for flow in flows:
                flow[np.abs(flow) < 0.5] = 0
            motions = map(optical_flow_to_motion, flows)
            motions = map(lambda x: transformer.preprocess('data', x), motions)
            predictions[idx] = motions_to_vec(motions, net)
            idx += 1
    return np.asarray(predictions)

def show_imgs(imgs):
    for img in imgs:
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

def write_features(predictions, video):
    with open(os.path.join(video, 'features' + '_' + SUFFIX + '.csv'), 'w') as f:
        np.savetxt(f, predictions, delimiter = ',')

def generate_features():
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST) 
    net.blobs['data'].reshape(1, 3, 227, 227)
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
    videos = map(lambda x: os.path.join(DATA_DIR, x), os.listdir(DATA_DIR))

    count = 1
    for video in videos:
        print '[INFO] processing video %d / %d' % (count, len(videos))
        predictions = video_to_vec(video, net, transformer)
        write_features(predictions, video)
        count += 1

def is_forward(video):
    video_name = filter(len, video.split('/'))[-1]
    if video_name[0] == 'F':
        return True
    return False

def load_features(video):
    with open(os.path.join(video, 'features' + '_' + SUFFIX + '.csv'), 'r') as f:
        X = np.loadtxt(f, delimiter = ',')
    return X

def load_labels(video):
    if is_forward(video):
        y = np.asarray([1, 1, -1, -1])
    else:
        y = np.asarray([-1, -1, 1, 1])
    return y

def load_features_labels(videos):
    X = np.asarray(map(load_features, videos))
    X = X.reshape(-1, X.shape[-1])
    y = np.asarray(map(load_labels, videos))
    y = y.reshape(1, -1).squeeze()
    return X, y

def load_dataset(train_list, test_list):
    X_train, y_train = load_features_labels(train_list)
    X_test, y_test = load_features_labels(test_list)
    return X_train, y_train, X_test, y_test

def load_list(dataset = 1, prefix = True):
    train_list = os.path.join(INDEX_DIR, 'train') + str(dataset) + '.txt'
    test_list = os.path.join(INDEX_DIR, 'test') + str(dataset) + '.txt'
    with open(train_list) as f:
        train_list = f.read().splitlines()
    with open(test_list) as f:
        test_list = f.read().splitlines()
    if prefix:
        train_list = map(lambda x: os.path.join(DATA_DIR, x), train_list)
        test_list = map(lambda x: os.path.join(DATA_DIR, x), test_list)
    return train_list, test_list

def predict_dataset(dataset = 1):
    """use naive SVM to predict
    """
    train_list, test_list = load_list(dataset)
    X_train, y_train, X_test, y_test_ = load_dataset(train_list, test_list)
    svc = SVC(kernel = 'rbf')
    svc.fit(X_train, y_train)
    y_predict_ = svc.predict(X_test)
    y_predict = np.empty(len(y_predict_) / 4)
    y_test = np.empty(len(y_predict))
    for i in range(len(y_predict)):
        k = 4*i
        y_predict[i] = y_predict_[k] + y_predict_[k+1] - y_predict_[k+2] - y_predict_[k+3]
        y_test[i] = y_test_[k]
    y_predict = np.sign(y_predict + 0.5)
    precision = np.sum(y_predict == y_test) / float(len(y_test))
    print 'Precision of dataset %d: %f' % (dataset, precision)
    return precision

def run_svm():
    map(predict_dataset, range(1, 4))

def generate_optical_flow():
    """generate optical flows of every video
    and save them to the same directory than
    the original video"""
    videos = os.listdir(DATA_DIR)
    for count in range(len(videos)):
        print '[INFO] processing video %d / %d' % (count, len(videos))
        video = videos[count]
        tmp_imgs = filter(lambda x: x[-4:] == 'jpeg', os.listdir(os.path.join(DATA_DIR, video)))
        tmp_orig_imgs = filter(lambda x: x[:2] == 'im', tmp_imgs)
        print len(tmp_imgs), len(tmp_orig_imgs)
        if (len(tmp_orig_imgs) - 2)*4+len(tmp_orig_imgs) == len(tmp_imgs):
            print 'already treated'
            continue
        fold = 1
        for reverse in [False, True]:
            for flip in [False, True]:
                imgs = load_video(os.path.join(DATA_DIR, video), reverse, flip)
                imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
                flows = optical_flow(imgs_gray)
                for flow in flows:
                    flow[np.abs(flow) < 0.5] = 0
                motions = map(optical_flow_to_motion, flows)
                for idx in range(len(motions)):
                    if os.path.isfile(os.path.join(DATA_DIR, video, 'of' + str(fold) + 'F' + str(idx+1).zfill(8) + '.jpeg')):
                        continue
                    motion = motions[idx]
                    if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                        cv2.imwrite(os.path.join(DATA_DIR, video, 'of' + str(fold) + 'F' + str(idx+1).zfill(8) + '.jpeg'), motion)
                    else:
                        cv2.imwrite(os.path.join(DATA_DIR, video, 'of' + str(fold) + 'B' + str(idx+1).zfill(8) + '.jpeg'), motion)
                fold += 1

if __name__ == '__main__':
    generate_optical_flow()
    #generate_features()
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
