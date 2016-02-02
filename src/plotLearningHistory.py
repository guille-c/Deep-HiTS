import sys
import numpy as np
import pylab as pl

import matplotlib
font = {'size'   : 14}

matplotlib.rc('font', **font)

train = np.load (sys.argv[1] + "/training_buffer_history.pkl")
valid = np.load (sys.argv[1] + "/validation_history.pkl")
test = 0.00839

i_tr = np.array([100] + train['iter_train_buf_history'])
e_tr = np.array([0.1818] + train['train_buf_err_history'])
#print i_tr
i_01 = i_tr[e_tr > 0.1]
print i_01[1:]-i_01[:-1]
print i_tr.shape, e_tr.shape

i_val = [100] + valid['iter_val_history']
e_val = [0.15783] + valid['val_err_history']

print train.keys()
print valid.keys()

print i_tr[0]
# crit_tr = []
# for i in i_tr:
    
log_values = 100*2**np.arange(0, np.log2(i_tr[-1]/100), 0.1)
print log_values
crit = (np.mod(i_tr, 2000) == 0)
crit = np.in1d(i_tr, log_values)
pl.clf()
#pl.plot (i_tr[crit], e_tr[crit], label = "Training set")
pl.plot (i_tr, e_tr, "k:", label = "Training set")
pl.plot (i_val, e_val, "k-", label = "Validation set", lw = 2)
#pl.axhline (test, ls = ":", label = "Test set", color = "k")
pl.scatter ([i_tr[-1]], [test], c = "b", marker = "x", s = 300, label = "Test set")
pl.legend (loc = "best", scatterpoints = 1)
pl.ylim([0, 0.05])
#pl.yscale("log")
pl.margins (x = 0.05)
pl.xlabel ("iteration")
pl.ylabel ("error")
pl.savefig (sys.argv[1] + "/learning_history.eps")

pl.clf()
pl.hist (e_tr, bins = 1000)
pl.savefig("hist_etrain")
