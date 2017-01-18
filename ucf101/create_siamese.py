import cv2
import lmdb
import caffe
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def write(df, of):
    env = lmdb.open(of, map_size = 1099511627776)
    batch_size = 50
    txn = env.begin(write=True)
    for index, row in df.iterrows():
	if index+1 % 100 == 0:
            print '[INFO] processing %d / %d' % (index+1, len(df))
        #prepare the data and label
        img1 = np.moveaxis(cv2.imread(row['img1']), -1, 0)
        img2 = np.moveaxis(cv2.imread(row['img2']), -1, 0)
        img3 = np.moveaxis(cv2.imread(row['img3']), -1, 0)
        image = np.concatenate([img1, img2, img3], axis = 0)
        # save in datum
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels, datum.height, datum.width = image.shape
        datum.data = image.tobytes()
        datum.label = row['label']
        str_id = '{:08}'.format(index)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        # write batch
        if (index+1) % batch_size == 0:
            print '[INFO] batch %d written' % index
            txn.commit()
            txn = env.begin(write=True)
    # write last batch
    if (index+1) % batch_size != 0:
        print '[INFO] last batch written'
        txn.commit()
    env.close()
if __name__ == '__main__':
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train['video'] = train['img1'].apply(lambda x: x.split('_')[1])
    test['video'] = test['img1'].apply(lambda x: x.split('_')[1])
    train['img1'] = 'UCF-101/' + train['video'] + '/' + train['img1'] + '.jpg'
    train['img2'] = 'UCF-101/' + train['video'] + '/' + train['img2'] + '.jpg'
    train['img3'] = 'UCF-101/' + train['video'] + '/' + train['img3'] + '.jpg'
    test['img1'] = 'UCF-101/' + test['video'] + '/' + test['img1'] + '.jpg'
    test['img2'] = 'UCF-101/' + test['video'] + '/' + test['img2'] + '.jpg'
    test['img3'] = 'UCF-101/' + test['video'] + '/' + test['img3'] + '.jpg'
    write(train, 'train')
    write(test, 'test')
