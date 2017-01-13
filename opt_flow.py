import os
import sys
import cv2
import lmdb
import caffe
import numpy as np
from multiprocessing.dummy import Pool
from helper import *

def registration(img1, img2):
    sz = img1.shape
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(img1,img2, warp_matrix)
        return cv2.warpAffine(img2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    except:
        return img2

def _optical_flow(imgs, i):
    img12 = registration(imgs[i], imgs[i-1])
    img23 = registration(imgs[i], imgs[i+1])
    flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)
    return flow

def optical_flow(imgs):
    imgs = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), imgs)
    p = Pool(3)
    flows = p.map(lambda x: _optical_flow(imgs, x), range(1,len(imgs) - 1))
    return flows

def run(videos, l = 0, u = 1000000):
    for video in videos[l:u]:
        print '[INFO] Processing video %d / %d' % (
                videos.index(video)+1, len(videos)
                )
        for reverse in [False, True]:
            imgs = load_video(video, './data/ArrowDataAll', reverse = reverse, mask = lambda x: x[:2] == 'im')
            flows = optical_flow(imgs)
            for count in range(len(flows)):
                flow = flows[count]
                flow_x = np.asarray(flow[..., 0], dtype = np.uint8)
                flow_y = np.asarray(flow[..., 1], dtype = np.uint8)
                if (is_forward(video) and (not reverse)) or ((not is_forward(video)) and reverse):
                    cv2.imwrite(os.path.join('./data/ArrowDataAll', video, 'off' + str(count+1).zfill(8) + 'x.jpeg'), flow_x)
                    cv2.imwrite(os.path.join('./data/ArrowDataAll', video, 'off' + str(count+1).zfill(8) + 'y.jpeg'), flow_y)
                else:
                    cv2.imwrite(os.path.join('./data/ArrowDataAll', video, 'ofb' + str(count+1).zfill(8) + 'x.jpeg'), flow_x)
                    cv2.imwrite(os.path.join('./data/ArrowDataAll', video, 'ofb' + str(count+1).zfill(8) + 'y.jpeg'), flow_y)
            
if __name__ == '__main__':
    l = int(sys.argv[1])
    r = int(sys.argv[2])
    train, test = load_list('./data')
    train.extend(test)
    train = filter(lambda x: not is_forward(x), train)
    run(train, l, r)
