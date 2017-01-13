import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
import lmdb

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

def forward_save(videos, lmdbfile):
    env = lmdb.open(lmdbfile, map_size = 1099511627776)
    idx = 0
    for video in videos:
        txn = env.begin(write = True)
        print '[INFO] processing video %d / %d' % (videos.index(video) + 1, len(videos))
	reverse = False
        if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
            direction = 'f'
        else:
            direction = 'b'
        imgs = np.asarray(load_video(video, './data/ArrowDataAll/', reverse = reverse, mask = lambda x: x[:2] == 'im'))
        flows = load_flows(video, 'f')
	sel = select(flows, 60)
	imgs = np.take(imgs, sel, axis = 0)
        imgs = map(lambda x: np.moveaxis(x, -1, 0), imgs)
        for i in range(len(imgs)/3):
            image = np.concatenate(imgs[3*i:3*(i+1)], axis = 0)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels, datum.height, datum.width = image.shape
            datum.data = image.tobytes()
            datum.label = 1 if direction == 'f' else 0
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            idx += 1
	    image = np.concatenate(imgs[3*i:3*(i+1)][::-1], axis = 0)
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels, datum.height, datum.width = image.shape
            datum.data = image.tobytes()
            datum.label = 0 if direction == 'f' else 1
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())
            idx += 1
        txn.commit()
    env.close()

if __name__ == '__main__':
    train_list, test_list = load_list('./data', dataset = 1)
    forward_save(train_list, './data/siamese-train1')
    forward_save(test_list, './data/siamese-test1')
