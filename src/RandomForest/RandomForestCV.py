import sys
import os
from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from time import time
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

sharedir = sys.argv[1]        # directory containing the feature files
prefix = "NOT TAKEN INTO ACCOUNT"       # prefix used in field naming convention, e.g. Blind14A for Blind14A_01, Blind14A_02, ...

K=     np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = bool)
diff = np.array([0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0], dtype = bool)
psf =  np.array([0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = bool)
snr=   np.array([0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0], dtype = bool)
i1=    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1], dtype = bool)
i2=    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1], dtype = bool)
ncand= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], dtype = bool)

#scd = SupernovaeCandidateData(sharedir, prefix, 2, 'lanczos2', shuffle = True)
#scd.printClasses()
#snrs = scd.attributes[:, 2]

pkl = np.load (sharedir + "/RF_train_set.pkl")
x_train = pkl["features"]
y_train = pkl["labels"]

pkl = np.load (sharedir + "/RF_test_set.pkl")
x_test = pkl["features"]
y_test = pkl["labels"]

pkl = np.load (sharedir + "/RF_val_set.pkl")
x_val = pkl["features"]
y_val = pkl["labels"]

print x_train.shape, y_train.shape
print x_test.shape, y_test.shape
print x_val.shape, y_val.shape

plotSNR = False
if plotSNR:
    import pylab as pl
    pl.clf()
    pl.hist (x_train[:, 2], bins = 50, range = [-30, 30])
    pl.xlabel ("SNR")
    pl.savefig ("SNRs")
    exit()

clf = RandomForestClassifier (random_state = 0, n_estimators=100,
                              criterion = 'entropy', max_features = 'auto')

N_test = 100000
N_train = 1000000
print N_train
i_s = np.arange(len(y_train))

np.random.shuffle (i_s)
x = x_train[i_s]
y = y_train[i_s]
n_iter = 6
scores, scores_s = [], []

sss = StratifiedShuffleSplit(y, n_iter=n_iter, test_size=N_test, train_size = N_train)
acc, prec, rec, f1 = np.zeros(n_iter), np.zeros(n_iter), \
                         np.zeros(n_iter), np.zeros(n_iter)
i = 0
for train_index, test_index in sss:
    t1 = time()
    clf.fit(x[train_index], y[train_index])
    y_p = clf.predict (x[test_index])

    fn = "RF_" + str(i) + ".pkl"
    with open (fn, "w") as f:
        pickle.dump({'RF_pbbs': y_p,
                     'labels': y[test_index],
                     'SNRs': x_train[test_index][:, 2]},
                    f, pickle.HIGHEST_PROTOCOL)

    acc[i] = accuracy_score (y[test_index], y_p)
    prec[i] = precision_score (y[test_index], y_p)
    rec[i] = recall_score (y[test_index], y_p)
    f1[i] = f1_score (y[test_index], y_p)

    t2 = time()
    print "   ", i, "total time = ", t2 - t1
    i += 1
    
print "acc = ", acc.mean(), "+-", acc.std()
print "prec = ", prec.mean(), "+-", prec.std()
print "rec = ", rec.mean(), "+-", rec.std()
print "f1 = ", f1.mean(), "+-", f1.std()

exit()
np.save(fn, np.array(accs))
scores.append (np.mean(accs))
scores_s.append (np.std(accs))

plot_acc = True
if plot_acc:
    import pylab as pl
    pl.clf()
    pl.errorbar (N_train, scores, yerr = scores_s, fmt = "-")
    pl.xscale("log")
    pl.xlim ([N_train[0]*0.7, N_train[-1]*2])
    pl.savefig ("acc_vs_Ntrain")
    #exit()
        
def trainRandomForest (x, y, x_t, y_t, clf, crit, fn, snrs):
    (att_train, att_test) = x[:, crit], x_t[:, crit]
    (class_train, class_test) = y, y_t
    
    print "fitting model, train = ", att_train.shape, ", test = ", att_test.shape
    clf.fit(att_train, class_train)
    print "model fitted, predicting"

    class_predict = clf.predict_proba(att_test)
    print "predicted, saving"

    with open (fn, "w") as f:
        pickle.dump({'RF_pbbs': class_predict,
                     'labels': class_test,
                     'SNRs': snrs},
                    f, pickle.HIGHEST_PROTOCOL)

# crit = np.ones(scd.attributes.shape[1])
# trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_all.pkl")
#crit = diff & np.logical_not(psf | snr | K | i1 | i2 | ncand)
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff.pkl")
#crit = (diff | psf) & np.logical_not(snr | K | i1 | i2 | ncand)
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff_psf.pkl")

#crit = (diff | psf | snr) & np.logical_not(K | i1 | i2 | ncand)
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff_psf_snr.pkl", snrs)
#crit = (diff | psf | snr | K) & np.logical_not(i1 | i2 | ncand)
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff_psf_snr_K.pkl", snrs)
#crit = (diff | psf | snr | K | i1 | i2) & np.logical_not(ncand)
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff_psf_snr_K_i1_i2.pkl", snrs)
#crit = diff | psf | snr | K | i1 | i2 | ncand
#trainRandomForest (scd, clf, crit, N_test, "RF_test_predictions_diff_psf_snr_K_i1_i2_cand.pkl", snrs)
crit = np.ones(len(diff), dtype = bool)
trainRandomForest (x_train, y_train, x_test, y_test, clf, crit,
                   sharedir + "/RF_test_predictions_full.pkl", x_test[:, 2])
