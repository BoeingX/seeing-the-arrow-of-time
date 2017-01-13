import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
from sklearn.svm import SVC

def show_imgs(imgs):
    for img in imgs:
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()

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

def find_optimal_batch(n):
    n_opt = 4
    i = 4
    while i <= min(n, 20):
        if n % i == 0:
            n_opt = i
        i += 4
    return n_opt

def load_flows(video, direction):
    imgs_ = load_video(video, './data/ArrowDataAll', 
                               mask = lambda x: x[:3] == 'of' + direction, 
                               grayscale=True)
    imgs_ = map(lambda x: cv2.resize(x, (227, 227)), imgs_)
    return imgs_
def select(imgs, n = 5):
    mags = [None] * (len(imgs)/2)
    for i in range(len(imgs)/2):
        mag, _ = cv2.cartToPolar(np.asarray(imgs[2*i], dtype = np.float32), np.asarray(imgs[2*i+1], dtype = np.float32))
    mags[i] = cv2.norm(mag)
    mags = np.asarray(mags)
    idx = np.argsort(mags)[::-1][:n]
    idx.sort()
    return idx

def forward_save(videos, net, transformer):
    for video in videos:
        X = np.empty((0, 4096*3))
        y = np.empty(0)
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
        for reverse in [False, True]:
            for flip in [False, True]:
                imgs = load_video(video, './data/ArrowDataAll/', flip = flip, reverse = reverse, mask = lambda x: x[:2] == 'im')
                if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                    direction = 'f'
                else:
                    direction = 'b'
                flows = load_flows(video, direction)
                idx = select(flows, 20)
                imgs = np.take(imgs, idx, axis = 0)
                imgs = map(lambda x: transformer.preprocess('data', x), imgs)
                for i in range(len(imgs) - 2):
                    net.blobs['data'].data[...] = imgs[i:i+3]
                    output = net.forward()
                    X = np.append(X, output['fc7'].flatten().reshape(1, -1), axis = 0)
                    if direction == 'f':
                        y = np.append(y, 1)
                    else:
                        y = np.append(y, 0)
        with open(os.path.join('./data/ArrowDataAll', video, 'features3.csv'), 'w') as f:
            np.savetxt(f, X, delimiter = ',', fmt = '%f')
        with open(os.path.join('./data/ArrowDataAll', video, 'labels3.csv'), 'w') as f:
            np.savetxt(f, y, delimiter = ',', fmt = '%d')

if __name__ == '__main__':
    #video = 'B1uglCr8ueyv8'
    #flip = True
    #reverse = False
    #imgs = load_video(video, './data/ArrowDataAll/', flip = flip, reverse = reverse)
    #if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
    #    direction = 'f'
    #else:
    #    direction = 'b'
    #print direction
    #flows = load_flows(video, direction)
    #idx = select(flows, 12)
    #imgs = np.take(imgs, idx, axis = 0)
    #show_imgs(imgs)

    caffe_root = '/home/bysong/caffe/'
    caffe.set_mode_gpu()
    # load original model
    model_def = 'models/caffenet3/deploy.prototxt'
    model_weights = 'models/caffenet3/bvlc_reference_caffenet.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST) 

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    
    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    net.blobs['data'].reshape(3, 3, 227, 227)

    train_list, test_list = load_list('./data', dataset = 1)
    videos = train_list
    videos.extend(test_list)
    forward_save(videos, net, transformer)
