#!/usr/bin/env python2
import cv2
import os
import numpy as np
import h5py
import lmdb
import caffe
from helper import *
def optical_flow(imgs):
    T = len(imgs)
    flows = []
    for i in range(1, T-1):
        flow = cv2.calcOpticalFlowFarneback(imgs[i-1],imgs[i+1], None, 0.5, 3, 15, 3, 5, 1.1, 0)
        flow[np.abs(flow) < 0.5] = 0
        flows.append(flow)
    return flows

def optical_flow_to_motion(flow):
    """convert optical 2-channel optical flow to RGB
    """
    #hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
    #hsv[...,1] = 255
    #mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    #hsv[...,0] = ang*180/np.pi/2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag

def generate_optical_flow(videos, h5name):
    """generate optical flows of every video
    and save them to the same directory than
    the original video"""
    data_dir = 'data/ArrowDataAll'
    idx = 0
    env = lmdb.open(h5name, map_size = int(1e12))
    with env.begin(write=True) as txn:
        for count in range(len(videos)):
            print '[INFO] processing video %d / %d' % (count, len(videos))
            video = videos[count]
            fold = 1
            for reverse in [False, True]:
                for flip in [False, True]:
                    imgs = load_video(video, data_dir, is_raw_img, reverse, flip)
                    imgs_gray = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
                    stacked_flow = []
                    flows = optical_flow(imgs_gray)
                    # select uniformly 50 frames
                    select = map(int, np.linspace(0, len(flows)-1, 10))
                    for s in select:
                        stacked_flow.append(flows[s][..., 0])
                        stacked_flow.append(flows[s][..., 1])
                    stacked_flow = np.asarray(stacked_flow)
                    if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                        label = 1
                    else:
                        label = 0
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = len(stacked_flow) 
                    datum.height = 224
                    datum.width = 224
                    datum.data = stacked_flow.tobytes()
                    datum.label = label 
                    str_id = '{:08}'.format(idx)
                    idx += 1
                    # The encode is only essential in Python 3
                    txn.put(str_id.encode('ascii'), datum.SerializeToString())
def run(dataset = 1):
    data_dir = './data'
    train_list, test_list = load_list(data_dir, dataset, False)
    generate_optical_flow(train_list, os.path.join(data_dir, 'train' + str(dataset)))
    generate_optical_flow(test_list, os.path.join(data_dir, 'test' + str(dataset)))

if __name__ == '__main__':
    run()
    #map(run, range(1, 4))
