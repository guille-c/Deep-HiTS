"""
Save misclassified candidates as PNG files.

Call it as:

>> python2 misclassified.pkl /saving/folder/
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

mis_dict = np.load(sys.argv[1])

mis_len = len(mis_dict['real_labels'])
n_selection = min(50, mis_len) # Number of candidates that we want to visualize
print n_selection, 'of', mis_len, 'candidates selected for visualization'
idx = np.random.permutation(mis_len)[:n_selection]

ims = mis_dict['misclassified_candidates'][idx]
preds = mis_dict['predicted_labels'][idx]
reals = mis_dict['real_labels'][idx]
snrs = mis_dict['SNRs'][idx]

im_size = 21
im_size_sq = im_size**2
count = 0
for candidate, pred, real, snr in zip(ims, preds, reals, snrs):
    temp_im = candidate[:im_size_sq].reshape([im_size, im_size])
    sci_im = candidate[im_size_sq:2*im_size_sq].reshape([im_size, im_size])
    diff_im = candidate[2*im_size_sq:3*im_size_sq].reshape([im_size, im_size])
    snr_im = candidate[3*im_size_sq:].reshape([im_size, im_size])
    stamps = [temp_im, sci_im, diff_im, snr_im]
    p_sn_model = pred[1]
    real = int(real)

    titles = ['Template', 'Science', 'Difference', 'SNR difference']
    title_aux_real = ['negative', 'positive']
    fig = plt.figure()
    fig.suptitle('Real class: '+title_aux_real[real]+', SN prob (model): '+
                 '%.3f'%(p_sn_model)+'. SNR=%.3f'%(snr), fontsize=16, fontweight='bold')
    for i, stamp in enumerate(stamps):
        plt.subplot(2,2,i+1)
        plt.imshow(stamp, interpolation='none')
        plt.set_cmap('summer')
        plt.axis('off')
        plt.title(titles[i])
    #plt.show()
    plt.savefig(sys.argv[2]+str(count).zfill(3)+'.png', bbox_inches='tight')
    plt.close()
    count += 1
    
