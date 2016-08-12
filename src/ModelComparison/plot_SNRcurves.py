import sys
import numpy as np
import pylab as pl
import cPickle as pickle
import argparse

from sklearn.metrics import roc_curve, auc
import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

do_ranges = True
pkl_RF  = np.load(sys.argv[1])
pkl_CNN = np.load(sys.argv[2])
print pkl_RF.keys()
print pkl_CNN.keys()

snr_thresholds = [7, 15]

if do_ranges:
    N_th = len(snr_thresholds) + 1
else:
    N_th = len(snr_thresholds)

colors = ['b', 'r', 'g', 'c', 'k']

pl.clf()
for i in range(N_th):
    #snrt = snr_thresholds[i]
    if do_ranges:
        if i == 0:
            crit_RF = (np.abs(pkl_RF['SNRs']) < snr_thresholds[i])
            crit_CNN = (np.abs(pkl_CNN['SNRs']) < snr_thresholds[i])
            lab = "SNR < " + str(snr_thresholds[i])
        elif i == len(snr_thresholds):
            crit_RF = (np.abs(pkl_RF['SNRs']) >= snr_thresholds[i-1])
            crit_CNN = (np.abs(pkl_CNN['SNRs']) >= snr_thresholds[i-1])
            lab = r"SNR $\geq$" + str(snr_thresholds[i-1])
        else:
            crit_RF = (np.abs(pkl_RF['SNRs']) >= snr_thresholds[i-1]) & (np.abs(pkl_RF['SNRs']) < snr_thresholds[i])
            crit_CNN = (np.abs(pkl_CNN['SNRs']) >= snr_thresholds[i-1]) & (np.abs(pkl_CNN['SNRs']) < snr_thresholds[i])
            lab = str(snr_thresholds[i-1]) + "$\leq$ SNR $<$" + str(snr_thresholds[i])
    else:
        crit_RF = (np.abs(pkl_RF['SNRs']) < snr_thresholds[i])
        crit_CNN = (np.abs(pkl_CNN['SNRs']) < snr_thresholds[i])
        lab = "SNR < " + str(snr_thresholds[i])
    lab += ", N = " + str(crit_RF.sum())
    print i, lab
    print crit_RF.sum(), crit_CNN.sum()
    y_test = pkl_RF['labels'][crit_RF]
    RF_test = pkl_RF['RF_pbbs'][crit_RF][:, 1]

    print y_test.shape, RF_test.shape, crit_RF.sum()
    fpr, tpr, _ = roc_curve(y_test, RF_test)
    pl.plot(fpr, 1.-tpr, label = "RF, " + lab, ls = "--", color = colors[i], lw = (3-i))
    
    #crit = (np.abs(pkl_CNN['SNRs']) < snrt)
    y_test = pkl_CNN['labels'][crit_CNN]
    CNN_test = pkl_CNN['ConvNet_pbbs'][crit_CNN][:, 1]

    print y_test.shape, CNN_test.shape, crit_CNN.sum()
    fpr, tpr, _ = roc_curve(y_test, CNN_test)
    pl.plot(fpr, 1.-tpr, label = "CNN, " + lab, ls = "-", color = colors[i], lw = (3-i))


pl.legend(loc="best", fontsize = "medium")
pl.xscale("log")
pl.yscale("log")
pl.xlabel("FPR")
pl.ylabel("FNR")
pl.xlim([1e-5, 3e-1])
pl.grid(True)
if do_ranges:
    pl.savefig("DET_SNR_ranges_curve.eps")
else:
    pl.savefig("DET_SNRcurve.eps")
    
pbbs = np.arange(0, 1, 1e-5)
fprs = []
fnrs = []
pbbs_plot = []
y = pkl_CNN['labels'][crit_CNN]
CNN = pkl_CNN['ConvNet_pbbs'][:, 1][crit_CNN]
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

