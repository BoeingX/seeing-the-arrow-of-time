# UCF-101 dataset

This is a collection of scripts which performs preprocessing of UCF-101 dataset to feed siamese network. Only tested under `GNU/Linux`.

## Preparation

One can download the full dataset from [here](http://crcv.ucf.edu/data/UCF101/UCF101.rar). 

A white list consisting frames used from training Siamese network can be found [here](https://s3.eu-central-1.amazonaws.com/mva-boeingx/ucf101/whitelist.txt)

`unrar`, `ffmpeg` and `GNU Parallel` should also be installed.

## Preprocessing

The following commands decompress archive, extract frames using `ffmpeg`, keep only frames from a `whitelist` and resize resting frames to 256x256.

```
    unrar x UCF101.rar
    find . -type f -name "*.avi" | parallel 'mkdir {.} && mv {} {.}/'
    find . -mindepth 2 -type d | parallel 'mkdir {}/images'
    find . -mindepth 2 -type f -name "*.avi" | parallel 'filename=$(basename {}); ffmpeg -loglevel "panic" -i {} -qscale:v 1 -f image2 {//}/images/"${filename}"_%06d.jpg && rm {}'
    find . -mindepth 2 -type f -name "*.jpg" | grep -f whitelist.txt -v | xargs rm
    find . -mindepth 2 -type f -name "*.jpg" | parallel 'mogrify -resize 256x256\! {}'
```

Be sure to have at **60GB** for above processing. 

## Train/val split

Train and validation triple names for Siamese network can be found [here](https://s3.eu-central-1.amazonaws.com/mva-boeingx/ucf101/train.csv) and [here](https://s3.eu-central-1.amazonaws.com/mva-boeingx/ucf101/test.csv).

## Create input for Siamese network

Launch `create_siamese.py` (make sure to modify paths accordingly).
