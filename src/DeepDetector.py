""" 
Call it as:
>> python2 DeepDetector.py arch7.py final_convnet_state_arch_7.pkl
"""
import os
import sys
import time
import imp

import math

import numpy
import cPickle as pickle
import theano
import theano.tensor as T


from layers import *
from loadHITS import *
from ChunkLoader import *

from ArchBuilder import ArchBuilder

from ConfigParser import ConfigParser

class ConvNet():
    def __init__(self, archpy_filename, batch_size=100,
                 activation=leaky_relu, params=[None]*100):
        rng = numpy.random.RandomState(23455)
        self.batch_size = batch_size
        im_chan = 4
        self.im_chan = im_chan
	
        # Creation of validation and test sets

        t_x = np.zeros([self.batch_size, 441*self.im_chan], dtype = th.config.floatX)
        t_y = np.zeros([self.batch_size], dtype = "int32")
        
        print "batch size: ", len(t_y)
        self.test_set_x, self.test_set_y = shared_dataset ([t_x, t_y])
    
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        
        im_size = 21
        self.input_data = self.x.reshape((batch_size, self.im_chan, im_size, im_size))
        print "input_data", (batch_size, self.im_chan, im_size, im_size)


        imp.load_source("archpy", archpy_filename)
        import archpy
        
        self.arch_def = archpy.convNetArchitecture(im_chan, batch_size, im_size, activation) 
        self.arch = ArchBuilder(self.arch_def, self.input_data, params, rng)
	self.layers = self.arch.getLayers()

	print "compiling 'predict' function"
        self.predict = theano.function([],self.layers[-1].output,
                                  givens={self.x: self.test_set_x},
                                  on_unused_input='ignore')

        print "compilation ready"
        DropoutLayer.deactivate()

    def load_params(self, params):
        self.arch.loadParams(params)
        
    def predict_sn(self, candidate):
	# 'candidate' has just one SNa candidate with c01 shape
	data = candidate.astype('float32')
        self.test_set_x.set_value(data)

	prediction = self.predict()
	return prediction

class DeepDetector:
    def __init__(self, archpy_filename, params_filename, batch_size, activation=leaky_relu):
	model = np.load(params_filename)
	params = model['best_params']	
	self.convnet = ConvNet(archpy_filename, params=params,
                               batch_size=batch_size,
                               activation=activation) # TODO
    
    def predict_sn(self, candidate):
	return self.convnet.predict_sn(candidate)
	
	
def normalize_stamp(stamp):
    ma = stamp.max()
    mi = stamp.min()
    return 1.0*(stamp-mi)/(ma-mi)
    
if __name__ == '__main__':
    n_examples = 1000
    print 'Using architecture from', sys.argv[1]
    print 'Loading params at file', sys.argv[2]
    deepDetector = DeepDetector(sys.argv[1], sys.argv[2], batch_size=n_examples)
    
    examples = np.load('/home/shared/HiTS_14A_chunks_2016_09/all/chunk_0_5000.pkl')
    temp_im = examples['temp_images']
    sci_im = examples['sci_images']
    diff_im = examples['diff_images']
    snr_im = examples['SNR_images']
    labels = examples['labels']
    
    ok_count = 0
    
    ims = []
    for ind in range(n_examples):
	i1 = normalize_stamp(temp_im[ind,:])
	i2 = normalize_stamp(sci_im[ind,:])
	i3 = normalize_stamp(diff_im[ind,:])
	i4 = normalize_stamp(snr_im[ind,:])
	candidate = np.concatenate((i1, i2, i3, i4), axis=0)
	#candidate = np.expand_dims(candidate, axis=0)
	ims.append(candidate)
    ims = np.asarray(ims)
    print ims.shape
    prediction = deepDetector.predict_sn(ims)
    ok_count += np.count_nonzero(np.equal(np.argmax(prediction, axis=1),labels[:n_examples]))
	
    print "Correct classifications: ", ok_count, "of", n_examples
    
    
