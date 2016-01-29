import sys
import numpy as np
import pylab as pl

train = np.load (sys.argv[1] + "/training_history.pkl")
valid = np.load (sys.argv[1] + "/validation_history.pkl")

i_tr = np.array(train['iter_train_history'])
e_tr = np.array(train['train_err_history'])
#print i_tr
i_01 = i_tr[e_tr > 0.1]
print i_01[1:]-i_01[:-1]
print i_tr.shape, e_tr.shape

i_val = valid['iter_val_history']
e_val = valid['val_err_history']

print train.keys()
print valid.keys()

crit = (np.mod(i_tr, 500) == 0)
pl.clf()
#pl.plot (i_tr[crit], e_tr[crit], label = "Training set")
pl.plot (i_tr, e_tr, label = "Training set")
pl.plot (i_val, e_val, label = "Validation set")
pl.legend (loc = "best")
pl.xlim([-1, 10000])
#pl.ylim([0, 0.02])
pl.savefig (sys.argv[1] + "/learning_history")
