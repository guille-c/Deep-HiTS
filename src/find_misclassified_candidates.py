"""
Call it as:

>> python2 find_misclassified_candidates.py arch7.py final_convnet_state.pkl /folder/with/chunks/ /saving/folder/misclassified.pkl
"""

import sys
import os

import numpy as np
import cPickle as pickle

from DeepDetector import *

assert len(sys.argv) == 5

batch_size = 5000
print 'Using architecture from', sys.argv[1]
print 'Loading params at file', sys.argv[2]
deepDetector = DeepDetector(sys.argv[1], sys.argv[2], batch_size=batch_size)

def get_normalized_chunk(chunk_dict, chunk_size=5000):
    temp_im = chunk_dict['temp_images']
    sci_im = chunk_dict['sci_images']
    diff_im = chunk_dict['diff_images']
    snr_im = chunk_dict['SNR_images']
    labels = chunk_dict['labels'][:chunk_size]
    snrs = chunk_dict['SNRs'][:chunk_size]
    ims = []
    for ind in range(chunk_size):
	i1 = normalize_stamp(temp_im[ind,:])
	i2 = normalize_stamp(sci_im[ind,:])
	i3 = normalize_stamp(diff_im[ind,:])
	i4 = normalize_stamp(snr_im[ind,:])
	candidate = np.concatenate((i1, i2, i3, i4), axis=0)
	ims.append(candidate)
    ims = np.asarray(ims)
    return (ims, labels, snrs)


misclassified_candidates = []
predicted_labels = []
real_labels = []
all_SNRs = []
for chunk_filename in os.listdir(sys.argv[3]):
    chunk_dict = np.load(sys.argv[3]+chunk_filename)
    ims, labels, snrs = get_normalized_chunk(chunk_dict, chunk_size=5000)
    prediction = deepDetector.predict_sn(ims)
    errors = np.not_equal(np.argmax(prediction, axis=1),labels)
    misclassified_candidates.append(ims[errors])
    predicted_labels.append(prediction[errors])
    real_labels.append(labels[errors])
    all_SNRs.append(snrs)
    print 100.0*np.count_nonzero(errors)/5000.0, '% error'
    
misclassified_candidates = np.concatenate(misclassified_candidates, axis=0)
predicted_labels = np.concatenate(predicted_labels, axis=0)
real_labels = np.concatenate(real_labels, axis=0)
all_SNRs = np.concatenate(all_SNRs, axis=0)

print len(real_labels), 'misclassified candidates on', sys.argv[3]
print int(real_labels.sum()), 'were false negatives'
print int(len(real_labels)-real_labels.sum()), 'were false positives'

with open(sys.argv[4], 'w') as f:
    pickle.dump({'misclassified_candidates': misclassified_candidates,
                 'predicted_labels': predicted_labels,
                 'real_labels': real_labels,
                 'SNRs': all_SNRs},
                f, pickle.HIGHEST_PROTOCOL)
