import os
import sys
from main import load_list, is_forward
WORKING_DIR = os.getcwd()
INDEX_DIR = os.path.join(WORKING_DIR, 'helper')
DATA_DIR = os.path.join(WORKING_DIR, 'data/ArrowDataAll')

def process_video(video):
    imgs = os.listdir(os.path.join(DATA_DIR, video))
    imgs = filter(lambda x: x[:2] == 'of' and (x[2] == '1' or x[2] == '3'), imgs)
    imgs = filter(lambda x: x[-6] == '5' or x[-6] == '0', imgs)
    imgs = map(lambda x: os.path.join(DATA_DIR, video, x), imgs)
    return imgs

def process_list(lst):
    imgs = map(process_video, lst)
    # 2d-list to 1d
    imgs = reduce(lambda x, y: x+y, imgs)
    return imgs

def run(dataset = 1):
    """train and test list are full paths
    """
    train_list, test_list = load_list(dataset, False)
    train_imgs = process_list(train_list)
    test_imgs = process_list(test_list)
    with open(os.path.join(WORKING_DIR, 'data', 'train' + str(dataset) + '.txt'), 'w') as f:
        for img in train_imgs:
            f.write(img)
            f.write(' ')
            if img[-14] == 'F':
                f.write('1')
            else:
                f.write('0')
            f.write('\n')
    with open(os.path.join(WORKING_DIR, 'data', 'test' + str(dataset) + '.txt'), 'w') as f:
        for img in test_imgs:
            f.write(img)
            f.write(' ')
            if img[-14] == 'F':
                f.write('1')
            else:
                f.write('0')
            f.write('\n')

if __name__ == '__main__':
    map(run, range(1, 4))
