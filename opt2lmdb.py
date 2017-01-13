
# coding: utf-8

# In[2]:

import os
import sys
import cv2
import lmdb
import caffe
import numpy as np


# In[3]:

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


# In[4]:

def load_flows(video):
    images = []
    labels = []
    for direction in ['f', 'b']:
        for flip in [False, True]:
            imgs_ = load_video(video, './data/ArrowDataAll', 
                               mask = lambda x: x[:3] == 'of' + direction, 
                               flip = flip, 
                               grayscale=True)
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


# In[5]:

def write_flows(video, env, baseidx):
    flows, labels = load_flows(video)
    txn = env.begin(write = True)
    for i in range(len(labels)):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 2
        datum.height = 256
        datum.width = 256
        datum.data = flows[i].tobytes()
        datum.label = labels[i]
        str_id = '{:08}'.format(i+baseidx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
    txn.commit()
    return len(labels) + baseidx


# In[6]:

train_list, test_list = load_list('./data', dataset = 1)


# In[ ]:

env = lmdb.open('data/train1', map_size = 1099511627776)
base_idx = 0
for video in train_list:
    print '[INFO] processing video %d / %d' % (train_list.index(video), len(train_list))
    base_idx = write_flows(video, env, base_idx)
env.close()


# In[ ]:

env = lmdb.open('data/test1', map_size = 1099511627776)
base_idx = 0
for video in test_list:
    print '[INFO] processing video %d / %d' % (test_list.index(video), len(test_list))
    base_idx = write_flows(video, env, base_idx)
env.close()

