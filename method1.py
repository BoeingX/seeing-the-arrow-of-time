
# coding: utf-8

# In[ ]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
from sklearn.svm import SVC
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

def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    width, height, _ = img.shape
    factor = max(256.0 / width, 256.0 / height)
    img = cv2.resize(img, None, fx = factor, fy = factor)
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

def select(imgs):
    mags = [None] * (len(imgs)/2)
    for i in range(len(imgs)/2):
        mag, _ = cv2.cartToPolar(np.asarray(imgs[2*i], dtype = np.float32), np.asarray(imgs[2*i+1], dtype = np.float32))
    mags[i] = cv2.norm(mag)
    mags = np.asarray(mags)
    idx = np.argsort(mags)[::-1][:10]
    indices = [None] * 20
    for i in range(10):
        indices[2*i] = 2*idx[i]
        indices[2*i+1] = 2*idx[i]+1
    image = np.take(np.asarray(imgs), indices, axis = 0)
    return image

def load_flows(video):
    images = []
    if is_forward(video):
        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'off')
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'off', flip = True)
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'ofb')
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))
        
        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'ofb', flip = True)
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        labels = [1, 1, 0, 0]
    else:
        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'ofb')
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))
        
        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'ofb', flip = True)
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'off')
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        imgs = load_video(video, './data/ArrowDataAll', mask = lambda x: x[:3] == 'off', flip = True)
        imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)
        images.append(select(imgs))

        labels = [0, 0, 1, 1]
    return np.asarray(images), np.asarray(labels)


# In[ ]:

caffe.set_mode_gpu()
# load original model
model_def = 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST) 
# mean the conv1 layer across the 3 channels
conv1_mean = net.params['conv1'][0].data.mean(axis = 1)
# duplicate 20 times to handle 20-channel input data
conv1_stack = np.repeat(conv1_mean[:, np.newaxis, :, :], 20, axis = 1)
                                                                                                 
# load modified model 
# with layers later than fc7 removed
model_def = 'models/caffenet/deploy.prototxt'
model_weights = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def, model_weights, caffe.TEST) 
# network surgery
net.params['conv1-stack'][0].data[...] = conv1_stack
net.blobs['data'].reshape(4, 20, 227, 227)


# In[ ]:

# load list 1
train_list, test_list = load_list('./data', dataset = 1)
X_train = np.empty((0, 4096))
y_train = np.empty(0)
for train in train_list:
    print '[INFO] processing video %d / %d' % (train_list.index(train) + 1, len(train_list))
    images, labels = load_flows(train)
    images -= 128*np.ones_like(images)
    output = net.forward()
    X_train = np.append(X_train, output['fc7'])
    y_train = np.append(y_train, labels)


# In[ ]:

X_test = np.empty((0, 4096))
y_test = np.empty(0)
for test in test_list:
    print '[INFO] processing video %d / %d' % (test_list.index(test) + 1, len(test_list))
    images, labels = load_flows(test)
    images -= 128*np.ones_like(images)
    output = net.forward()
    X_test = np.append(X_test, output['fc7'])
    y_test = np.append(y_test, labels)


# In[ ]:

svc = SVC(kernel = 'rbf')
svc.fit(X_train, y_train)


# In[ ]:

print 'raw precision %f' % svc.score(X_test, y_test)


# In[ ]:

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
return precisionsvc.score(X_test, y_test)


# In[ ]:



