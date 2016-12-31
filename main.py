#!/usr/bin/env python2
from preprocessing import *
import pandas as pd
import caffe

CAFFE_ROOT = os.path.join(os.path.expanduser('~'), 'caffe')
MODEL_FILE = os.path.join(CAFFE_ROOT, 'models/binarygooglenet/deploy.prototxt')
PRETRAINED = os.path.join(CAFFE_ROOT, 'binarygooglenet1_iter_1368.caffemodel')


## method 1
def predict_dataset(dataset = 1):
    """use naive SVM to predict
    """
    train_list, test_list = load_list('./data/', dataset, False)
    data_dir = './data/ArrowDataAll/'
    X_train, y_train, X_test, y_test_ = load_dataset(train_list, test_list, data_dir, 'bvlc_reference_caffenet')
    svc = SVC(kernel = 'rbf')
    svc.fit(X_train, y_train)
    y_predict_ = svc.predict(X_test)
    y_predict = np.empty(len(y_predict_) / 4)
    y_test = np.empty(len(y_predict))
    for i in range(len(y_predict)):
        k = 4*i
        y_predict[i] = y_predict_[k] + y_predict_[k+1] - y_predict_[k+2] - y_predict_[k+3]
        y_test[i] = y_test_[k]
    y_predict = np.sign(y_predict + 0.5)
    precision = np.sum(y_predict == y_test) / float(len(y_test))
    print 'Precision of dataset %d: %f' % (dataset, precision)
    return precision

def run_svm():
    map(predict_dataset, range(1, 4))

## method 2
def predict_img(img, net):
    net.blobs['data'].data[...] = img
    output = net.forward()
    output_prob = output['prob']
    return output_prob.argmax()

def load_optical_flow(filename):
    img = cv2.imread(filename)
    return img

def predict(filename, transformer, net):
    print filename
    img = load_optical_flow(filename)
    img = transformer.preprocess('data', img)
    return predict_img(img, net)

def run_caffe(dataset = 1):
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST) 
    net.blobs['data'].reshape(1, 3, 224, 224)
    mu = np.load(CAFFE_ROOT + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mu)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    with open('./data/test1.txt') as f:
        test = pd.read_csv(f, delimiter = ' ', header = None)
    test[3] = test.apply(lambda x: predict(x[0], transformer, net), axis = 1)
    test.to_csv('test1.csv', sep = ' ', header = False)

if __name__ == '__main__':
    run_caffe()
