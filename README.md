# Seeing the arrow of time

This is the code repository for course project `Seeing the arrow of time` which consists of determining the direction of time of video being played.

## Dependencies

-   `OpenCV 3` (See troubleshooting for using `OpenCV2`)
-   `Caffe`

## Preparation

### YouTube dataset

The dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/arrow/ArrowDataAll.tgz). It is suppose to decompressed to `data` folder.

Three train-val splits can be found [here](https://s3.eu-central-1.amazonaws.com/mva-boeingx/youtube/train-val.tar.gz). Those files are suppose to be decompressed to `data` folder.

Launch `opt_flow.py` to generate optical flows for the dataset.

### UCF-101

C.f. [UCF-101](ucf101/README.md). 

## Troubleshooting

-   Use `OpenCV 2` instead of `OpenCV 3`

Change `flow = cv2.calcOpticalFlowFarneback(img12, img23, None, 0.5, 3, 15, 3, 5, 1.2, 0)` to `flow = cv2.calcOpticalFlowFarneback(img12, img23, 0.5, 3, 15, 3, 5, 1.2, 0)` in `opt_flow.py`.
