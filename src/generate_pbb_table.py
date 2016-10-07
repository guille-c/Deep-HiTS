import os
import sys

import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

from DeepDetector import *

def load_data(chunks_dir):
    if not os.path.isdir(chunks_dir):
        raise Exception(chunks_dir+' is not a valid directory')
    new_dict = dict()
    keys = ['features',
            'temp_images',
            'sci_images',
            'diff_images',
            'SNR_images',
            'SNRs',
            'labels']
    for key in keys:
        new_dict[key] = []
    for chunk_filename in os.listdir(chunks_dir):
        print chunk_filename
        pkl = np.load(chunks_dir+chunk_filename)
        if len(pkl['labels'])<5000:
            continue
        for key in keys:
            new_dict[key].append(pkl[key][:5000])
    for key in keys:
        new_dict[key] = np.concatenate(new_dict[key], axis=0)
    new_dict['id'] = np.arange(len(new_dict['labels']))
    return new_dict

def calculate_accuracy(labels, pbbs):
    model_is_sne = pbbs>0.5 # setting 0.5 as standard threshold
    model_class = model_is_sne.astype(np.float)
    return np.equal(labels, model_class).astype(float).mean()

#data = load_data('/home/shared/Fields_12-2015/chunks_feat_5000/chunks_test/')
#with open('/home/shared/RF_5000/test_set.pkl', 'w') as f:
#    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

data = np.load('/home/shared/RF_5000/test_set.pkl')
print 'Data loaded'

# Calculate RF pbbs
with open('/home/shared/RF_5000/RF_model.pkl', 'rb') as fid:
    rf_model = pickle.load(fid)
print 'Random Forest model loaded'
rf_pbbs = rf_model.predict_proba(data['features'])[:,1]
print 'Pbbs calculated (Random Forest)'

print 'Random Forest accuracy', calculate_accuracy(data['labels'], rf_pbbs)

# Calculate DL pbbs
batch_size = 5000

##### TO DO: ACTIVATION IS NOT BEING READ FROM arch.py, SO IT HAS TO BE PASSED BY HAND
#from layers import * 

deepDetector = DeepDetector('/home/shared/arch7/arch7.py',
                            '/home/shared/arch7/final_convnet_state_32bits.pkl',
                            batch_size=batch_size)
    
temp_im = data['temp_images']
sci_im = data['sci_images']
diff_im = data['diff_images']
snr_im = data['SNR_images']

ims = []
for ind in range(len(data['labels'])):
    i1 = normalize_stamp(temp_im[ind,:])
    i2 = normalize_stamp(sci_im[ind,:])
    i3 = normalize_stamp(diff_im[ind,:])
    i4 = normalize_stamp(snr_im[ind,:])
    candidate = np.concatenate((i1, i2, i3, i4), axis=0)
    ims.append(candidate)
ims = np.asarray(ims).astype(np.float32)
print ims.shape

n_samples = len(data['labels'])
n_batches = n_samples // batch_size
convnet_pbbs = []
for i in range(n_batches):
    start_idx = i*batch_size
    print 'Calculating ConvNet probabilities:', start_idx
    end_idx = start_idx + batch_size
    convnet_pbbs.append(deepDetector.predict_sn(ims[start_idx:end_idx])[:,1])
convnet_pbbs = np.concatenate(convnet_pbbs, axis=0)

print 'Convolutional Neural Network accuracy', calculate_accuracy(data['labels'], convnet_pbbs)

print data['labels'][:10]
print convnet_pbbs[:10]

csv_data = np.asarray([data['id'],
                           data['labels'],
                           data['SNRs'],
                           rf_pbbs,
                           convnet_pbbs]).transpose(1,0)
print csv_data.shape
np.savetxt('/home/shared/RF_5000/rf_cnn_pbbs.csv', csv_data, delimiter=',')


def saveGifs(ims, indexes, folder):
    im_size = 21
    im_size_sq = im_size**2
    titles = ['Template', 'Science', 'Difference', 'SNR difference']
    for candidate, index in zip(ims, indexes):
        temp_im = candidate[:im_size_sq].reshape([im_size, im_size])
        sci_im = candidate[im_size_sq:2*im_size_sq].reshape([im_size, im_size])
        diff_im = candidate[2*im_size_sq:3*im_size_sq].reshape([im_size, im_size])
        snr_im = candidate[3*im_size_sq:].reshape([im_size, im_size])
        stamps = [temp_im, sci_im, diff_im, snr_im]

        fig = plt.figure()
        for i, stamp in enumerate(stamps):
            plt.subplot(2,2,i+1)
            plt.imshow(stamp, interpolation='none')
            plt.set_cmap('gray')
            plt.axis('off')
            plt.title(titles[i])
        #plt.show()
        plt.savefig(folder+str(index)+'.png', bbox_inches='tight')
        plt.close()

saveGifs(ims[:10], data['id'][:10], '/home/shared/RF_5000/GIFs/')
