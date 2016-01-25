import sys
import os
import glob
import cPickle as pickle
import numpy as np

def loadDirectory (dirname, chunk_size = 5000):
    if not os.path.exists (dirname):
        print "ERROR, ", dirname, " does not exist"
    files = glob.glob (dirname + "chunk*.pkl")
    x, y = np.array([]).reshape((0, 56)), np.array([])
    for f in files:
        pkl = np.load(f)
        print f, len(pkl["labels"])
        if len(pkl["labels"]) < chunk_size:
            continue
        x1 = pkl["features"][:chunk_size]
        y1 = pkl["labels"][:chunk_size]
        # bad chunk
        #if f == "chunk_95_5000.pkl":
        #    x1 = np.vstack ((x1[:3500], x1[4000:]))
        #    y1 = np.concatenate ((y1[:3500], y1[4000:]))
        x = np.vstack((x, x1))
        y = np.concatenate((y, y1))
    return x, y
#    exit()
    
sharedir = sys.argv[1]        # directory containing the chunk files
chunk_size = int(sys.argv[3])

# K=     np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = bool)
# diff = np.array([0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0], dtype = bool)
# psf =  np.array([0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = bool)
# snr=   np.array([0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0], dtype = bool)
# i1=    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,1,0,0,0,0,0,0,0,0,1], dtype = bool)
# i2=    np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1], dtype = bool)
# ncand= np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], dtype = bool)

x_train, y_train = loadDirectory (sharedir + "/chunks_train/",
                                  chunk_size = chunk_size)
print x_train.shape, y_train.shape
with open(sys.argv[2] + "/RF_train_set.pkl", "w") as f:
    pickle.dump({"features": x_train,
                 "labels": y_train},
                f, pickle.HIGHEST_PROTOCOL)

x_test, y_test = loadDirectory (sharedir + "/chunks_test/",
                                chunk_size = chunk_size)
print x_test.shape, y_test.shape
with open(sys.argv[2] + "/RF_test_set.pkl", "w") as f:
    pickle.dump({"features": x_test,
                 "labels": y_test},
                f, pickle.HIGHEST_PROTOCOL)

x_valid, y_valid = loadDirectory (sharedir + "/chunks_validate/",
                                  chunk_size = chunk_size)
print x_valid.shape, y_valid.shape
with open(sys.argv[2] + "/RF_val_set.pkl", "w") as f:
    pickle.dump({"features": x_valid,
                 "labels": y_valid},
                f, pickle.HIGHEST_PROTOCOL)
