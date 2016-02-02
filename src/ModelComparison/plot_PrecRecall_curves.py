import sys
import numpy as np
import pylab as pl
from sklearn.metrics import auc, roc_curve, precision_recall_fscore_support, accuracy_score
from sklearn.metrics.classification import _check_targets
#precision_recall_curve, average_precision_score

import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

in_file = open(sys.argv[1])

pl.clf()
ls = [":", "--", "-", "-."]
i = 0
for line in in_file:
    fn, pred_name, lab = line.split(";")
    
    pkl = np.load (fn)

    y = pkl["labels"]
    pred = pkl[pred_name][:, 1]

    #prec_RF, rec, _ = precision_recall_curve(y_RF, pred_RF)
    fpr, tpr, _ = roc_curve(y, pred)
    auc_ = auc(fpr, tpr)

    #pl.plot (1-rec_RF, 1-prec_RF, label = ", AUC = {0:0.2f}".format(av_RF))
    #pl.plot (fpr, 1-tpr, label = lab[:-1] + ", AUC = {0:0.3f}".format(auc_))
    pl.plot (fpr, 1-tpr, label = lab[:-1], ls = ls[i], c = "k")

    prec, rec, f1, supp = precision_recall_fscore_support(y, (pred >= 0.5), average='binary')
    acc = accuracy_score (y, (pred >= 0.5))
    print lab[:-1], " & ", np.round(acc, 4), " & ", np.round(prec, 4), " & ", np.round(rec, 4), " & ", np.round(f1, 4), "\\\\"
    i += 1
    
pl.xlabel("FPR")
pl.ylabel("FNR")
pl.xscale("log")
pl.yscale("log")
pl.legend(loc = "lower left")
pl.savefig ("DET_models.eps")

