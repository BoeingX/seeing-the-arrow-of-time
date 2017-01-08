import os
import sys
import cv2
import lmdb
import caffe
import numpy as np
from multiprocessing.dummy import Pool

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

def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    img = cv2.resize(img, (256, 256))
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

def optical_flow(imgs):
    imgs = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
    T = len(imgs)
    flows = [] 
    for i in range(T-1):
        flow = cv2.calcOpticalFlowFarneback(imgs[i],imgs[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)
    return flows

def _optical_flow_to_motion(flow):
    """convert optical 2-channel optical flow to RGB
    """
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag = np.asarray(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX), dtype = np.uint8)
    return mag

def optical_flow_to_motion(flows):
    p = Pool(4)
    mags = p.map(_optical_flow_to_motion, flows)
    return mags

def extract_key_frames(imgs, n_frames = 3):
    flows = optical_flow(imgs)
    motions = optical_flow_to_motion(flows)
    max_mags = np.argsort(np.asarray(map(cv2.norm, motions)))
    take = np.sort(max_mags[:n_frames])
    frames = np.take(imgs, take, axis = 0)
    return frames

def is_forward(video):
    if video[0] == 'F':
        return True
    return False

def stack_img(imgs):
    assert len(imgs) == 3
    stacked_imgs = []
    for img in imgs:
        stacked_imgs.append(img[...,0])
        stacked_imgs.append(img[...,1])
        stacked_imgs.append(img[...,2])
    return np.asarray(stacked_imgs, dtype = np.uint8)

def run(videos, lmdbfile, limit = 1000000):
    env = lmdb.open(lmdbfile, map_size = int(1e12))
    txn = env.begin(write = True)
    idx = 0
    batch_size = 30
    for video in videos[:limit]:
        print video
        imgs = load_video(video, './data/ArrowDataAll')
        frames = extract_key_frames(imgs, 5)

        print 'forward'
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[:3])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 1
        else:
            datum.label = 0
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[1:4])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 1
        else:
            datum.label = 0
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[2:5])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 1
        else:
            datum.label = 0
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        print 'backward'
        frames = frames[::-1]
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[:3])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 0
        else:
            datum.label = 1
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[1:4])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 0
        else:
            datum.label = 1
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 9
        datum.height = 256
        datum.width = 256
        stacked_imgs = stack_img(frames[2:5])
        datum.data = stacked_imgs.tobytes()
        if is_forward(video):
            datum.label = 0
        else:
            datum.label = 1
        str_id = '{:08}'.format(idx)
        idx += 1
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        print idx
        if(idx) % batch_size == 0:
            txn.commit()
            txn = env.begin(write=True)
            print 'batch %d written' % idx
    txn.commit()
    env.close()
if __name__ == '__main__':
    train, test = load_list('./data')
    #run(train, './data/train1')
    run(test, './data/test1')
