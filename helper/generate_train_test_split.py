#!/usr/bin/env python2
import os
import sys
from random import shuffle

if __name__ == '__main__':
    try:
        data_dir = sys.argv[1]
    except:
        data_dir = '../data/ArrowDataAll'
    try:
        target_dir = sys.argv[2]
    except:
        target_dir = './'
    data = [d for d in os.listdir(data_dir)]
    forward_all = filter(lambda x: x[0] == 'F', data)
    backward_all = filter(lambda x: x[0] == 'B', data)
    print '[INFO] %d forward videos' % len(forward_all)
    print '[INFO] %d backward videos' % len(backward_all)

    print '[INFO] shuffling video names'
    shuffle(forward_all)
    shuffle(backward_all)

    forward_split = [52, 52, 51]
    backward_split = [8, 8, 9]

    forward = [None] * 3
    backward = [None] * 3
    forward[0] = forward_all[:52]
    forward[1] = forward_all[52:104]
    forward[2] = forward_all[104:]

    backward[0] = backward_all[:8]
    backward[1] = backward_all[8:16]
    backward[2] = backward_all[16:]

    print '[INFO] forward videos partitioned into %d, %d, %d' % \
        (len(forward[0]), len(forward[1]), len(forward[2]))
    print '[INFO] backward videos partitioned into %d, %d, %d' % \
        (len(backward[0]), len(backward[1]), len(backward[2]))


    print '[INFO] generating train1.txt...'
    print '[INFO] F:B = %d:%d' % (len(forward[0]) + len(forward[1]), len(backward[0]) + len(backward[1]))
    with open(os.path.join(target_dir, 'train%d.txt' % 1), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[0])
        map(lambda x: f.write(str(x) + '\n'), forward[1])
        map(lambda x: f.write(str(x) + '\n'), backward[0])
        map(lambda x: f.write(str(x) + '\n'), backward[1])

    print '[INFO] generating test1.txt'
    print '[INFO] F:B = %d:%d' % (len(forward[2]), len(backward[2]))
    with open(os.path.join(target_dir, 'test%d.txt' % 1), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[2])
        map(lambda x: f.write(str(x) + '\n'), backward[2])

    print '[INFO] generating train2.txt...'
    print '[INFO] F:B = %d:%d' % (len(forward[1]) + len(forward[2]), len(backward[1]) + len(backward[2]))
    with open(os.path.join(target_dir, 'train%d.txt' % 2), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[1])
        map(lambda x: f.write(str(x) + '\n'), forward[2])
        map(lambda x: f.write(str(x) + '\n'), backward[1])
        map(lambda x: f.write(str(x) + '\n'), backward[2])

    print '[INFO] generating test2.txt'
    print '[INFO] F:B = %d:%d' % (len(forward[0]), len(backward[0]))
    with open(os.path.join(target_dir, 'test%d.txt' % 2), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[0])
        map(lambda x: f.write(str(x) + '\n'), backward[0])

    print '[INFO] generating train3.txt...'
    print '[INFO] F:B = %d:%d' % (len(forward[0]) + len(forward[2]), len(backward[0]) + len(backward[2]))
    with open(os.path.join(target_dir, 'train%d.txt' % 3), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[0])
        map(lambda x: f.write(str(x) + '\n'), forward[2])
        map(lambda x: f.write(str(x) + '\n'), backward[0])
        map(lambda x: f.write(str(x) + '\n'), backward[2])

    print '[INFO] generating test3.txt'
    print '[INFO] F:B = %d:%d' % (len(forward[1]), len(backward[1]))
    with open(os.path.join(target_dir, 'test%d.txt' % 3), 'w') as f:
        map(lambda x: f.write(str(x) + '\n'), forward[1])
        map(lambda x: f.write(str(x) + '\n'), backward[1])
