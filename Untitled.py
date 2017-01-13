
# coding: utf-8

# # Presentation

# In[1]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
import os
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
# In[2]:

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

def _save_flow(flow, filename):
    flow_x = np.asarray(flow[..., 0], dtype = np.uint8)
    flow_y = np.asarray(flow[..., 1], dtype = np.uint8)
    cv2.imwrite(filename + '_x.jpeg', flow_x)
    cv2.imwrite(filename + '_y.jpeg', flow_y)


# In[3]:

video = 'F_aqvxyejK0MQ'
imgs = load_video(video, 'data/ArrowDataAll/', mask = lambda x: x[:2] == 'im')


# In[4]:

img12 = registration(imgs[1], imgs[0])


# In[5]:

img23 = registration(imgs[1], imgs[2])


# In[6]:

img12 = cv2.cvtColor(img12, cv2.COLOR_BGR2GRAY)
img23 = cv2.cvtColor(img23, cv2.COLOR_BGR2GRAY)


# In[7]:

flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)
flow = cv2.normalize(flow, None, 0, 255, cv2.NORM_MINMAX)


# In[8]:

flowx = flow[..., 0]
flowy = flow[..., 1]


# In[9]:

plt.subplot(1, 3, 1)
plt.imshow(imgs[0])
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(imgs[1])
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(imgs[2])
plt.axis('off')
plt.tight_layout()
plt.savefig('../report/figs/opt-flow-1.pdf')


# In[10]:

plt.subplot(1, 2, 1)
plt.imshow(img12)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img23)
plt.axis('off')
plt.tight_layout()
plt.savefig('../report/figs/opt-flow-2.pdf')


# In[11]:

plt.subplot(1, 2, 1)
plt.imshow(flowx)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(flowy)
plt.axis('off')
plt.tight_layout()
plt.savefig('../report/figs/opt-flow-3.pdf')


# In[ ]:



