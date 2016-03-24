import os
import sys
import time

import numpy
import cPickle as pickle
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from loadHITS import *

class ChunkLoader():
    def __init__(self, folder, n_cand_chunk, batch_size, n_rot = 0):
	self.files = os.listdir(folder)
        self.files.sort()
	self.current_file = 0
	self.batch_i = 0
        self.folder = folder
        self.n_cand_chunk = n_cand_chunk
        self.batch_size = batch_size
	self.current_file_data = np.load(self.folder+self.files[self.current_file])
	self.lastSNRs = []
        self.done = False
        self.n_rot = n_rot
        
    def normalizeImage(self, im):
	return 1. * (im - im.min())/(im.max() - im.min())
	
    def normalizeSet(self, data):
	for i in range(len(data)):
	    data[i] = self.normalizeImage(data[i])
	return data

    def current_minibatch_SNR(self):
        return self.lastSNRs
    
    def nextFile (self):
	self.batch_i = 0
	self.current_file = (self.current_file+1)%len(self.files)
        self.current_file_data = np.load(self.folder + self.files[self.current_file])
        if self.current_file == 0:
            self.done = True
            
    def getNext(self, normalize=True):
	#print self.current_file, self.batch_i, self.files[self.current_file]
	keys = ['SNR_images']

	#N = len(train_pkl['labels'])
        self.lastSNRs = self.current_file_data['SNRs'][self.batch_i:self.batch_i+self.batch_size]
	data = []
	for k in keys:
	    temp = self.current_file_data[k][self.batch_i:self.batch_i+self.batch_size]
	    if normalize:
	        temp = self.normalizeSet(temp)
	    data.append(temp)
	if self.n_rot > 0:
	    for i in range (len(data)):
		data.append(rot90(data[i]))    # 90 degrees
                if self.n_rot > 1:
                    data.append(rot90(data[-1]))   # 180 degrees
                if self.n_rot > 2:
                    data.append(rot90(data[-1]))   # 270 degrees

	data = np.array(data, dtype = "float32")
	data = np.swapaxes(data, 0, 1)
	s = data.shape
	data = data.flatten().reshape((s[0], s[1]*s[2]))
	labels = np.array(self.current_file_data['labels'][self.batch_i:self.batch_i+self.batch_size], dtype="int32")
	train_set = [data, labels]
	self.batch_i += self.batch_size
        if train_set[0].shape[0] < self.batch_size:
            print "ERROR: ", self.folder + self.files[self.current_file], " has ", train_set[0].shape[0], " candidates."
            self.nextFile()
            return self.getNext (normalize)

	if self.batch_i+self.batch_size>self.n_cand_chunk:
            self.nextFile()

	return train_set
	    
if __name__=="__main__":
    c = ChunkLoader('/home/ireyes/chunks_feat_50000/chunks_train/', 50000, 50, n_rot=0)
    n_epochs = 3
    for e in np.arange(n_epochs):
        print "e = ", e
        total = 0
        while not c.done:
	    x,y = c.getNext()
            total += len(y)
	    print e, total#x.shape, y.shape
        c.done = False
