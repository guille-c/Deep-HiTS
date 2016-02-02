import sys
import numpy as np
import pylab as pl
import cPickle as pickle
import argparse

from sklearn.metrics import roc_curve, auc
import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

pkl_RF  = np.load(sys.argv[1])
pkl_CNN = np.load(sys.argv[2])
print pkl_RF.keys()
print pkl_CNN.keys()

snr_thresholds = [7, 10, 100]

colors = ['b', 'r', 'g', 'c', 'k']

pl.clf()
for i in range(len(snr_thresholds)):
    snrt = snr_thresholds[i]
    crit = (np.abs(pkl_RF['SNRs']) < snrt)
    y_test = pkl_RF['labels'][crit]
    RF_test = pkl_RF['RF_pbbs'][crit][:, 1]

    print y_test.shape, RF_test.shape, crit.sum()
    fpr, tpr, _ = roc_curve(y_test, RF_test)
    pl.plot(fpr, 1.-tpr, label = "RF SNR < " + str(snrt) + ', AUC = ' + str(np.round(1-auc(fpr, tpr), 3)), ls = "--", color = colors[i], lw = (3-i))
    
    crit = (np.abs(pkl_CNN['SNRs']) < snrt)
    y_test = pkl_CNN['labels'][crit]
    CNN_test = pkl_CNN['ConvNet_pbbs'][crit][:, 1]

    print y_test.shape, CNN_test.shape, crit.sum()
    fpr, tpr, _ = roc_curve(y_test, CNN_test)
    pl.plot(fpr, 1.-tpr, label = "ConvNet-4 SNR < " + str(snrt) + ', AUC = ' + str(np.round(1-auc(fpr, tpr), 3)), ls = "-", color = colors[i], lw = (3-i))


pl.legend(loc="best", fontsize = "medium")
pl.xscale("log")
pl.yscale("log")
pl.xlabel("FPR")
pl.ylabel("FNR")
pl.xlim([1e-5, 3e-1])
pl.savefig("DET_SNRcurve.eps")

pbbs = np.arange(0, 1, 1e-5)
fprs = []
fnrs = []
pbbs_plot = []
y = pkl_CNN['labels']
CNN = pkl_CNN['ConvNet_pbbs'][:, 1]
P = (y == 1).sum()
N = (y == 0).sum()
print "P = ", P
print "N = ", N
for pbb in pbbs:
    FN = ((CNN < pbb) & (y >= pbb)).sum()
    FP = ((CNN >= pbb) & (y < pbb)).sum()
    fprs.append(1.*FP/N) # % of negatives detected as positives
    fnrs.append(1.*FN/P) # % of positives missed

pl.clf()
pl.plot(pbbs, fprs, label = "FPR")
pl.plot(pbbs, fnrs, label = "FNR")
pl.legend()
pl.xlabel("probability")
pl.yscale("log")
pl.margins(x = 0.1)
pl.savefig ("FPR_FNR_vs_pbb")

