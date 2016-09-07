""" Architecture 7 / Model F with He initialization

Call it as:
>> python2 DeepDetector.py final_convnet_state_arch_7.pkl
"""
import os
import sys
import time

import math

import numpy
import cPickle as pickle
import theano
import theano.tensor as T


from layers import *
from loadHITS import *
from ChunkLoader import *

from ConfigParser import ConfigParser

class ArchBuilder():
    def __init__(self, arch, input_data, params, rng):
        self.params = params
        self.i_params = 0
        self.rng = rng
        self.layers = []
        dataInfo = arch[0]
        print dataInfo
        assert dataInfo["layer"]=="DataLayer"
        arch = arch[1:]# push out DataLayer
        input_shape = [dataInfo["batch_size"],
                      dataInfo["im_chan"],
                      dataInfo["im_size"],
                      dataInfo["im_size"]]
        self.layers.append(DataLayer(input_data, input_shape))
        for layerInfo in arch:
            self.addLayer(layerInfo)

    def addLayer(self, layerInfo):
        layerType = layerInfo["layer"]
        input_shape = self.layers[-1].getOutputShape()
        print "input_shape: ", input_shape
        print layerInfo
        if layerType=="CreateRotationsLayer":
            self.layers.append(CreateRotationsLayer(self.layers[-1].output,
                                                    input_shape))
        elif layerType=="ConvPoolLayer":
            filter_shape = [layerInfo["num_output"],
                            input_shape[1],
                            layerInfo["filter_size"],
                            layerInfo["filter_size"]
            ]
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(ConvPool2Layer(self.rng,
					      input=self.layers[-1].output,
                                              filter_shape=filter_shape,
                                              image_shape=input_shape,
                                              pad=layerInfo["pad"],
                                              poolsize=(layerInfo["pool_size"],
                                                        layerInfo["pool_size"]),
                                              activation = layerInfo["activation"],
                                              poolstride=(layerInfo["pool_size"],
                                                          layerInfo["pool_size"]),
                                              init_type=layerInfo["init_type"],
                                              W=W,
                                              b=b))
        elif layerType=="JoinRotationsLayer":
            self.layers.append(JoinRotationsLayer(self.layers[-1].output,
                                                  input_shape))
        elif layerType=="HiddenLayer":
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(HiddenLayer(
                self.rng,
                input=self.layers[-1].output,
                n_in=input_shape[1],
                n_out=layerInfo["num_output"],
                activation=layerInfo["activation"],
                init_type=layerInfo["init_type"],
                W=W,
                b=b))
            self.layers[-1].setBatchSize(input_shape[0])
        elif layerType=="DropoutLayer":
            self.layers.append(DropoutLayer(self.layers[-1].output,
                                            p_drop = layerInfo["p_drop"]))
            self.layers[-1].setOutputShape(input_shape)
        elif layerType=="LogisticRegression":
            W = self.params[self.i_params]
            self.i_params += 1
            b = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(LogisticRegression(input=self.layers[-1].output,
                                                  n_in=input_shape[1],
                                                  n_out=layerInfo["num_output"],
                                                  init_type=layerInfo["init_type"],
                                                  W=W,
                                                  b=b,
                                                  rng=self.rng))
            self.layers[-1].setBatchSize(input_shape[0])
        elif layerType=="ResidualLayer":
            filter_shape = [layerInfo["num_output"],
                            input_shape[1],
                            layerInfo["filter_size"],
                            layerInfo["filter_size"]
            ]
            W1 = self.params[self.i_params]
            self.i_params += 1
            b1 = self.params[self.i_params]
            self.i_params += 1
            W2 = self.params[self.i_params]
            self.i_params += 1
            b2 = self.params[self.i_params]
            self.i_params += 1
            self.layers.append(ResidualLayer(rng=self.rng,
                                             input=self.layers[-1].output,
                                             filter_shape=filter_shape,
                                             image_shape=input_shape,
                                             activation=layerInfo["activation"],
                                             W1 = W1,
                                             b1 = b1,
                                             W2 = W2,
                                             b2 = b2))                                        
        elif layerType=="MaxPoolingLayer":
            self.layers.append(MaxPoolingLayer(input=self.layers[-1].output,
                                               image_shape = input_shape,
                                               size = layerInfo["size"],
                                               stride = layerInfo["stride"]))                           
        else:
            raise Exception("layerType "+str(layerType)+" is not valid")
        print "Output shape ",self.layers[-1].getOutputShape()
        print "-----------------"
    def getLayers(self):
        return self.layers
    def getParams(self):
        params = []
        for layer in self.layers:
            if layer.hasParams():
                params = params+layer.params
        return params
    def loadParams(self, params):
        i = 0
        for layer in self.layers:
            nParams = layer.nParams()
            if nParams==2:
                layer.load_params(params[i],params[i+1])
                i+=2
            elif nParams==4:
                layer.load_params(params[i],
                                  params[i+1],
                                  params[i+2],
                                  params[i+3])
                i+=4
            elif nParams==0:
                pass
            else:
                raise Exception("nParams==",nParams)
                

class ConvNet():
    def __init__(self, batch_size=100,
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

        self.arch_def = [{"layer": "DataLayer",
                          "im_chan": im_chan,
                          "batch_size": batch_size,
                          "im_size": im_size},
                         {"layer": "CreateRotationsLayer"},
                         {"layer": "ConvPoolLayer",
                          "num_output": 32,
                          "filter_size": 4,
                          "pad": 3,
                          "pool_size": 1,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "ConvPoolLayer",
                          "num_output": 32,
                          "filter_size": 3,
                          "pad": 1,
                          "pool_size": 2,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "ConvPoolLayer",
                          "num_output": 64,
                          "filter_size": 3,
                          "pad": 1,
                          "pool_size": 1,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "ConvPoolLayer",
                          "num_output": 64,
                          "filter_size": 3,
                          "pad": 1,
                          "pool_size": 1,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "ConvPoolLayer",
                          "num_output": 64,
                          "filter_size": 3,
                          "pad": 1,
                          "pool_size": 2,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "JoinRotationsLayer"},
                         {"layer": "HiddenLayer",
                          "num_output": 64,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "DropoutLayer",
                          "p_drop": 0.5},
                         {"layer": "HiddenLayer",
                          "num_output": 64,
                          "activation": activation,
                          "init_type": "ReLU"},
                         {"layer": "DropoutLayer",
                          "p_drop": 0.5},
                         {"layer": "LogisticRegression",
                          "num_output": 2,
                          "init_type": "ReLU"}]
        
                     
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
    def __init__(self, params_filename, batch_size):
	model = np.load(params_filename)
	params = model['best_params']	
	self.convnet = ConvNet(params=params, batch_size=batch_size) # TODO
    
    def predict_sn(self, candidate):
	return self.convnet.predict_sn(candidate)
	
	
def normalize_stamp(stamp):
    ma = stamp.max()
    mi = stamp.min()
    return 1.0*(stamp-mi)/(ma-mi)
    
if __name__ == '__main__':
    n_examples = 1000
    
    print 'Loading params at file', sys.argv[1]
    deepDetector = DeepDetector(sys.argv[1], batch_size=n_examples)
    
    examples = np.load('/home/shared/Fields_12-2015/chunks_feat_50000/all_chunks/chunk_0_50000.pkl')
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
    
    
