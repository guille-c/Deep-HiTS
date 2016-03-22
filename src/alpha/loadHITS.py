import numpy as np
import theano as th
import theano.tensor as T
import os
import sys
import glob
import cPickle as pickle

def normalizeImage (im):
    return 1. * (im - im.min())/(im.max() - im.min())

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = th.shared(np.asarray(data_x,
                                    dtype=th.config.floatX),
                         borrow=borrow)
    shared_y = th.shared(np.asarray(data_y,
#                                    dtype=th.config.floatX),
                                    dtype="int32"),
                         borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    #print type (shared_x), type (shared_y)
    #return shared_x, T.cast(shared_y, 'int32')
    return shared_x, shared_y

def loadHITSCCD (dir, Field, CCD, n_stamp = 21, i_y = 10, i_RF = 12, i_SNR = 4):
    files = glob.glob (dir + Field + "/" + CCD + "/CANDIDATES/cand_" + Field + "_" + CCD
                       + "*_grid02_lanczos2.npy")
    y = np.array([])
    pbb_RF = np.array([])
    snr = np.array([])
    im_diffs = []
    im_scis = []
    im_temps = []
    im_SNRs = []
    for f in files:
        sim2, sim1 = f.split("_grid02_lanczos2.npy")[0].split(CCD + "_")[-1].split("-")
        feat_fn = dir + Field + "/" + CCD + "/CANDIDATES/features_"+ Field + "_" + CCD \
                  + "_" + sim2 + "-" + sim1 + "_grid02_lanczos2.npy"
        print f, feat_fn
        if not os.path.exists (feat_fn):
            print "ERROR loadHITSCCD: " + feat_fn + " doesnt exist."
            continue
            #exit()

        cand = np.load (f)
        feat = np.load(feat_fn)
        print cand.shape, feat.shape
        if cand.shape[0] != feat.shape[0]:
            print "ERROR: loadHITSCCD: features and candidates have different shapes:", \
                cand.shape[0], feat.shape[0]
            continue
            #exit()

        for icand in range(cand.shape[0]):
            imSNR = cand[icand, 12: 12 + n_stamp * n_stamp]
            im1 = cand[icand, 12 + n_stamp * n_stamp: 12 + 2 * n_stamp * n_stamp]
            im2 = cand[icand, 12 + 2 * n_stamp * n_stamp: 12 + 3 * n_stamp * n_stamp]
            imt = cand[icand, 12 + 3 * n_stamp * n_stamp: 12 + 4 * n_stamp * n_stamp]
            if (sim1[-1] == "t"):
                imdiff = im2 - imt
                im_temps.append(imt)
                im_scis.append(im2)
            else:
                imdiff = imt - im1
                im_temps.append(im1)
                im_scis.append(imt)

            im_SNRs.append(imSNR)
            # imSNR = normalizeImage(imSNR)
            # im1 = normalizeImage(im1)
            # im2 = normalizeImage(im2)
            # imt = normalizeImage(imt)
    
            im_diffs.append(imdiff)
            
        y = np.concatenate((y, feat[:, i_y]))
        pbb_RF = np.concatenate((pbb_RF, feat[:, i_RF]))
        snr = np.concatenate((snr, feat[:, i_SNR]))
    return np.array(im_temps), np.array(im_scis), np.array(im_diffs), np.array(im_SNRs), y, pbb_RF, snr
    
def loadHITSField (dir, Field, n_stamp = 21, i_y = 10, i_RF = 12):
    files = os.listdir (dir + Field)
    print files

    y = np.array([])
    pbb_RF = np.array([])
    snr = np.array([])
    im_ts = np.array([]).reshape((0, 441))
    im_ss = np.array([]).reshape((0, 441))
    im_ds = np.array([]).reshape((0, 441))
    im_snrs = np.array([]).reshape((0, 441))
    for f in files:
        #print dir, Field, f
        im_t, im_s, im_d, im_snr, y1, rf1, snr1 = loadHITSCCD (dir, Field, f)
        #new_im = np.zeros ((im_diffs.shape[0], im_diffs.shape[1], 3))
        #new_im[:, :, 0] = im_diffs
        print dir, Field, f, im_t.shape
        im_ts = np.vstack((im_ts, im_t))
        im_ss = np.vstack((im_ss, im_s))
        im_ds = np.vstack((im_ds, im_d))
        im_snrs = np.vstack((im_snrs, im_snr))
        y = np.concatenate((y, y1))
        pbb_RF = np.concatenate((pbb_RF, rf1))
        snr = np.concatenate((snr, snr1))
    return im_ts, im_ss, im_ds, im_snrs, y, pbb_RF, snr

def loadHITSObservation (dir, n_stamp = 21, i_y= 10, i_RF = 12):
    fields = glob.glob (dir + "Blind*")
    im_ts = np.array([]).reshape((0, 441))
    im_ss = np.array([]).reshape((0, 441))
    im_ds = np.array([]).reshape((0, 441))
    im_snrs = np.array([]).reshape((0, 441))
    ys = np.array([])
    pbbs_RF = np.array([])
    snrs = np.array([])
    for field in fields:
        f = field.split("/")[-1]
        im_t, im_s, im_d, im_snr, y, pbb_RF, snr = loadHITSField (dir, f, n_stamp,
                                         i_y, i_RF)
        im_ts = np.vstack((im_ts, im_t))
        im_ss = np.vstack((im_ss, im_s))
        im_ds = np.vstack((im_ds, im_d))
        im_snrs = np.vstack((im_snrs, im_snr))
        ys = np.concatenate((ys, y))
        pbbs_RF= np.concatenate((pbbs_RF, pbb_RF))
        snrs = np.concatenate((snrs, snr))
        print f, len(y), len(ys)
    return im_ts, im_ss, im_ds, im_snrs, ys, pbbs_RF, snrs

def loadHITSObservationTheano (dir, n_stamp = 21, i_y= 10, i_RF = 12,
                               N_train = False, N_valid = False,
                               N_test = False):

    diffs, ys, pbbs_RF, snrs = loadHITSObservation (dir, n_stamp, i_y, i_RF)
    
    i_s = np.arange(y.shape[0])
    np.random.shuffle(i_s)
    if not N_train:
        N_train = int(np.round(0.6 * y.shape[0]))
    if not N_valid:
        N_valid = int(np.round(0.2 * y.shape[0]))
    if not N_test:
        N_test = y.shape[0] - N_train - N_valid
    print N_train, N_valid, N_test

    i_train = i_s[:N_train]
    i_valid = i_s[N_train:N_train + N_valid]
    i_test  = i_s[N_train + N_valid:N_train + N_valid + N_test]
    
    train_set = [imts[i_train], ys[i_train]]
    valid_set = [imts[i_valid], ys[i_valid]]
    test_set  = [imts[i_test], ys[i_test]]
    
    print len(train_set), len(valid_set), len(test_set)
    print len(train_set[0]), len(valid_set[0]), len(test_set[0])
    print len(train_set[0][0]), len(valid_set[0][0]), len(test_set[0][0])
    print min(train_set[0][0]), max(train_set[0][0])


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval, [pbbs_RF[i_train], pbbs_RF[i_valid], pbbs_RF[i_test]]

def normalizeSet(data_set):
    for i in range (len(data_set['diff_images'])):
        data_set['diff_images'][i] = normalizeImage(data_set['diff_images'][i])

def rot90 (images):
    ret = []
    for im in images:
        ret.append(np.rot90 (im.reshape((21, 21))).flatten())
    return np.array(ret)
    
def loadHITSPklTheano (train_fn, test_fn, N_valid = False, normalize = True, rotate = True):
    keys = ['temp_images', 'sci_images', 'diff_images', 'SNR_images']
        
    train_pkl = np.load(train_fn)
    print train_pkl.keys()
    test_pkl = np.load(test_fn)
    if normalize:
        normalizeSet(train_pkl)
        normalizeSet(test_pkl)
    
        N = len(train_pkl['labels'])
    if not N_valid:
        N_valid = int(np.round(0.2 * N))

    data = []
    data_t = []
    for k in keys:
        data.append(train_pkl[k])
        data_t.append(test_pkl[k])
    if rotate:
        for i in range (len(data)):
            data.append(rot90(data[i]))    # 90 degrees
#            data.append(rot90(data[-1]))   # 180 degrees
#            data.append(rot90(data[-1]))   # 270 degrees

            data_t.append(rot90(data_t[i]))  # 90 degrees
#            data_t.append(rot90(data_t[-1])) # 180 degrees
#            data_t.append(rot90(data_t[-1])) # 270 degrees

    data = np.array(data)
    data_t = np.array(data_t)
    
    #data = np.array([train_pkl['temp_images'], train_pkl['sci_images'], train_pkl['diff_images'], train_pkl['SNR_images']])
    data = np.swapaxes(data, 0, 1)
    s = data.shape
    data = data.flatten().reshape((s[0], s[1]*s[2]))
    train_set = [data[N_valid:], train_pkl['labels'][N_valid:]]
    valid_set = [data[:N_valid], train_pkl['labels'][:N_valid]]

    #data_t = np.array([test_pkl['temp_images'], test_pkl['sci_images'], test_pkl['diff_images'], test_pkl['SNR_images']])
    #data_t = np.array([test_pkl['diff_images']])
    data_t = np.swapaxes(data_t, 0, 1)
    s = data_t.shape
    print s
    print s[0], s[1], s[2]
    data_t = data_t.flatten().reshape((s[0], s[1]*s[2]))
    test_set = [data_t, test_pkl['labels']]
    
    #train_set = [train_pkl['diff_images'][N_valid:], train_pkl['labels'][N_valid:]]
    #valid_set = [train_pkl['diff_images'][:N_valid], train_pkl['labels'][:N_valid]]
    #test_set  = [test_pkl['diff_images'], test_pkl['labels']]
    
    print len(train_set), len(valid_set), len(test_set)
    print len(train_set[0]), len(valid_set[0]), len(test_set[0])
    print len(train_set[0][0]), len(valid_set[0][0]), len(test_set[0][0])
    #print min(train_set[0][0]), max(train_set[0][0])
    print train_set[0].shape, test_set[0].shape

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval, [train_pkl['RF_pbbs'][N_valid:], train_pkl['RF_pbbs'][:N_valid],
                  test_pkl['RF_pbbs']]

def readTXTFile (filename):
    file = open (filename, "r")
    imts = []
    y = []
    for line in file:
        s = line.split(",")
        a = np.array(s[:-1], dtype = float)
        imts.append(normalizeImage(a[:441]))
        y.append(int(np.round(float(s[-1][:-1]))))
    return imts, y

def loadTXT (dir_in):
    x_train, y_train = readTXTFile (dir_in + "train.txt")
    x_valid, y_valid = readTXTFile (dir_in + "val.txt")
    x_test, y_test = readTXTFile (dir_in + "test.txt")

    train_set = [x_train, y_train]
    valid_set = [x_valid, y_valid]
    test_set = [x_test, y_test]

    print len(train_set), len(valid_set), len(test_set)
    print len(train_set[0]), len(valid_set[0]), len(test_set[0])
    print len(train_set[0][0]), len(valid_set[0][0]), len(test_set[0][0])
    print min(train_set[0][0]), max(train_set[0][0])

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

#diffs, y, pRF, snrs = loadHITSObservation (sys.argv[1])
#print diffs.shape, y.shape, pRF.shape, diffs[0].shape
