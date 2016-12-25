#!/usr/bin/env python2
import os
import cv2
def load_img(filename, flip = False):
    img = cv2.imread(filename)
    if flip:
        img = cv2.flip(img, 1)
    return img

def load_imgs(filenames, reverse = False, flip = False):
    if reverse:
        filenames = filenames[::-1]
    imgs = map(lambda x: load_img(x, flip), filenames)
    return imgs

def load_video(filename, reverse = False, flip = False):
    filenames = os.listdir(filename)
    filenames.sort()
    filenames = map(lambda x: video + x, filenames)
    imgs = load_imgs(filenames, True, False)
    return imgs

if __name__ == '__main__':
    video = "./data/ArrowDataAll/F_aqvxyejK0MQ/"
    imgs = load_video(video, True, False)
    for img in imgs:
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()
