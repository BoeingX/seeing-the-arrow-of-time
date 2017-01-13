
# coding: utf-8

# # Classification: Instant Recognition with Caffe
# 
# In this example we'll classify an image with the bundled CaffeNet model (which is based on the network architecture of Krizhevsky et al. for ImageNet).
# 
# We'll compare CPU and GPU modes and then dig into the model to inspect features and the output.

# ### 1. Setup
# 
# * First, set up Python, `numpy`, and `matplotlib`.

# In[2]:

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().magic('matplotlib inline')

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


# * Load `caffe`.

# In[3]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/bysong/caffe/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.


# * If needed, download the reference model ("CaffeNet", a variant of AlexNet).

# In[4]:

import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    get_ipython().system('../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')


# ### 2. Load net and set up input preprocessing
# 
# * Set Caffe to CPU mode and load the net from disk.

# In[11]:

caffe.set_mode_cpu()

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net_ = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# In[13]:

caffe.set_mode_cpu()

model_def = caffe_root + 'models/caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this, but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
# 
#     Our default CaffeNet is configured to take images in BGR format. Values are expected to start in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them. In addition, the channel dimension is expected as the first (_outermost_) dimension.
#     
#     As matplotlib will load images with values in the range [0, 1] in RGB format with the channel as the _innermost_ dimension, we are arranging for the needed transformations here.

# In[14]:

tmp = net_.params['conv1'][0].data.mean(axis = 1)
tmp.shape


# In[16]:

net.params['conv1-stack'][0].data[...] = np.repeat(tmp[:, np.newaxis, :, :], 20, axis = 1)


# In[63]:

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR


# ### 3. CPU classification
# 
# * Now we're ready to perform classification. Even though we'll only classify one image, we'll set a batch size of 50 to demonstrate batching.

# In[17]:

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          20,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227


# * Load an image (that comes with Caffe) and perform the preprocessing we've set up.

# In[18]:

import cv2
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
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if flip:
        img = cv2.flip(img, 1)
    width, height = img.shape
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


# In[19]:

data_dir = 'data/ArrowDataAll/'
video = 'F_aqvxyejK0MQ'
imgs = load_video(video, data_dir, mask = lambda x: x[:3] == 'off')
imgs = map(lambda x: cv2.resize(x, (227, 227)), imgs)


# In[20]:

plt.imshow(imgs[0])


# In[21]:

mags = [None] * (len(imgs)/2)
for i in range(len(imgs)/2):
    mag, _ = cv2.cartToPolar(np.asarray(imgs[2*i], dtype = np.float32), np.asarray(imgs[2*i+1], dtype = np.float32))
    mags[i] = cv2.norm(mag)
mags = np.asarray(mags)


# In[22]:

idx = np.argsort(mags)[::-1][:10]
indices = [None] * 20
for i in range(10):
    indices[2*i] = 2*idx[i]
    indices[2*i+1] = 2*idx[i]+12
image = np.take(np.asarray(imgs), indices, axis = 0)


# In[23]:

print image.shape


# * Adorable! Let's classify it!

# In[24]:

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


# * The net gives us a vector of probabilities; the most probable class was the 281st one. But is that correct? Let's check the ImageNet labels...

# In[9]:

# load ImageNet labels
labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    get_ipython().system('../data/ilsvrc12/get_ilsvrc_aux.sh')
    
labels = np.loadtxt(labels_file, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]


# * "Tabby cat" is correct! But let's also look at other top (but less confident predictions).

# In[10]:

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])


# * We see that less confident predictions are sensible.

# ### 4. Switching to GPU mode
# 
# * Let's see how long classification took, and compare it to GPU mode.

# In[11]:

get_ipython().magic('timeit net.forward()')


# * That's a while, even for a batch of 50 images. Let's switch to GPU mode.

# In[12]:

caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward()  # run once before timing to set up memory
get_ipython().magic('timeit net.forward()')


# * That should be much faster!

# ### 5. Examining intermediate output
# 
# * A net is not just a black box; let's take a look at some of the parameters and intermediate activations.
# 
# First we'll see how to read out the structure of the net in terms of activation and parameter shapes.
# 
# * For each layer, let's look at the activation shapes, which typically have the form `(batch_size, channel_dim, height, width)`.
# 
#     The activations are exposed as an `OrderedDict`, `net.blobs`.

# In[13]:

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


# * Now look at the parameter shapes. The parameters are exposed as another `OrderedDict`, `net.params`. We need to index the resulting values with either `[0]` for weights or `[1]` for biases.
# 
#     The param shapes typically have the form `(output_channels, input_channels, filter_height, filter_width)` (for the weights) and the 1-dimensional shape `(output_channels,)` (for the biases).

# In[14]:

for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


# * Since we're dealing with four-dimensional data here, we'll define a helper function for visualizing sets of rectangular heatmaps.

# In[15]:

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')


# * First we'll look at the first layer filters, `conv1`

# In[16]:

# the parameters are a list of [weights, biases]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))


# * The first layer output, `conv1` (rectified responses of the filters above, first 36 only)

# In[17]:

feat = net.blobs['conv1'].data[0, :36]
vis_square(feat)


# * The fifth layer after pooling, `pool5`

# In[18]:

feat = net.blobs['pool5'].data[0]
vis_square(feat)


# * The first fully connected layer, `fc6` (rectified)
# 
#     We show the output values and the histogram of the positive values

# In[19]:

feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)


# * The final probability output, `prob`

# In[20]:

feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)


# Note the cluster of strong predictions; the labels are sorted semantically. The top peaks correspond to the top predicted labels, as shown above.

# ### 6. Try your own image
# 
# Now we'll grab an image from the web and classify it using the steps above.
# 
# * Try setting `my_image_url` to any JPEG image URL.

# In[ ]:

# download an image
my_image_url = "..."  # paste your URL here
# for example:
# my_image_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Orang_Utan%2C_Semenggok_Forest_Reserve%2C_Sarawak%2C_Borneo%2C_Malaysia.JPG"
get_ipython().system('wget -O image.jpg $my_image_url')

# transform it and copy it into the net
image = caffe.io.load_image('image.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data', image)

# perform classification
net.forward()

# obtain the output probabilities
output_prob = net.blobs['prob'].data[0]

# sort top five predictions from softmax output
top_inds = output_prob.argsort()[::-1][:5]

plt.imshow(image)

print 'probabilities and labels:'
zip(output_prob[top_inds], labels[top_inds])

