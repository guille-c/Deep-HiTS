"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import math

import numpy
import cPickle as pickle
import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, DropoutLayer
from loadHITS import *
from ChunkLoader import *

from ConfigParser import ConfigParser

def rectify(x, alpha=0):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * abs(x)
def relu(x):
    return rectify(x, alpha=0)
    #return T.nnet.relu(x)
    #return T.switch(x<0, 0, x)
def prelu(x):
    return rectify(x, alpha=0.01)
    #return T.nnet.relu(x, alpha=0.01)
    return T.switch(x<0, 0.01*x, x)

class CreateRotationsLayer():
    def __init__(self, input, input_shape):
        # Sizes as bc01
        self.input_shape = input_shape
        assert self.input_shape[2]==self.input_shape[3]# image should be a square
        im_size = self.input_shape[2]
        self.output_shape = self.input_shape
        self.output_shape[0] = self.output_shape[0]*4
        self.part0 = input
        self.part1 = input[:,:,:,:-im_size-1:-1].dimshuffle(0,1,3,2) # 90 degrees
        self.part2 = input[:,:,:-im_size-1:-1,:-im_size-1:-1] # 180 degrees
        self.part3 = input[:,:,:-im_size-1:-1,:].dimshuffle(0,1,3,2) # 270 degrees

        self.output = T.concatenate([self.part0, self.part1, self.part2, self.part3], axis=0)

    def getOutputShape(self):
        return self.output_shape
    def hasParams(self):
        return False
    def nParams(self):
        return 0
    
class JoinRotationsLayer():
    def __init__(self, input, input_shape):
        # sizes as bc01
        self.input_shape = input_shape
        batch_size = self.input_shape[0]//4
        self.data_aux = input.reshape((4, batch_size, self.input_shape[1]*self.input_shape[2]*self.input_shape[3]))
        self.output = self.data_aux.transpose(1, 0, 2).reshape((batch_size, 4*self.input_shape[1]*self.input_shape[2]*self.input_shape[3]))
        self.output_shape = [batch_size, 4*self.input_shape[1]*self.input_shape[2]*self.input_shape[3]]

    def getOutputShape(self):
        return self.output_shape
    def hasParams(self):
        return False
    def nParams(self):
        return 0
    
class DataLayer():
    def __init__(self, input_data, input_shape):
        self.output = input_data
        self.output_shape = input_shape

    def getOutputShape(self):
        return self.output_shape
    def hasParams(self):
        return False
    def nParams(self):
        return 0

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
            self.layers.append(LeNetConvPoolLayer(self.rng,
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
                

class ResidualLayer():
    ### CHANGE WEIGHT INIT TO HE ET. AL. 2014 ###
    def __init__(self, rng, input, filter_shape, image_shape,
                 activation=prelu, W1=None, W2=None, b1=None, b2=None):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if W1==None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W1 = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W1 = W1

        if b1==None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b1 = theano.shared(value=b_values, borrow=True)
        else:
            self.b1 = b1


        assert filter_shape[2]%2==1# odd size
        pad = (filter_shape[2]-1)//2
            
        input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        w1_shuffled = self.W1.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op1 = FilterActs(stride=1, partial_sum=1, pad=pad)
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_w1 = gpu_contiguous(w1_shuffled)
        conv_out_1_shuffled = conv_op1(contiguous_input, contiguous_w1)
        conv_out_1 = conv_out_1_shuffled.dimshuffle(3, 0, 1, 2)# c01b to bc01
        activ_1_out = activation(conv_out_1+self.b1.dimshuffle('x',0,'x','x'))


        filter_shape[1] = filter_shape[0]
        if W2==None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
            # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W2 = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.W2 = W2

        if b2==None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b2 = theano.shared(value=b_values, borrow=True)
        else:
            self.b2 = b2

        w2_shuffled = self.W2.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        activ_1_out_shuffled = activ_1_out.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        contiguous_activ_1_out = gpu_contiguous(activ_1_out_shuffled)
        contiguous_w2 = gpu_contiguous(w2_shuffled)
        conv_op2 = FilterActs(stride=1, partial_sum=1, pad=pad)
        conv_out_2_shuffled = conv_op2(contiguous_activ_1_out, contiguous_w2)
        conv_out_2 = conv_out_2_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
        self.output = activation(conv_out_2+self.b2.dimshuffle('x', 0, 'x', 'x')+input)

        stride = 1# not used
        assert (image_shape[2]-filter_shape[2]+2*pad)%stride==0
        output_im_size = (image_shape[2]-filter_shape[2]+2*pad)/stride+1
        self.output_shape = [image_shape[0],
                            filter_shape[0],
                            output_im_size,
                            output_im_size]
                            
        # store parameters of this layer
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def getOutputShape(self):
        return self.output_shape
    
    def load_params(self, W1, b1, W2, b2):
        self.W1.set_value(W1)
        self.b1.set_value(b1)
        self.W2.set_value(W2)
        self.b2.set_value(b2)

        print 'Residual Layer parameters loaded'
    def hasParams(self):
        return True
    def nParams(self):
        return 4

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,
                 pad = 0, poolsize=(2, 2), activation = T.tanh, poolstride=(2, 2),
                 init_type="tanh",
                 W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        if W==None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = numpy.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            
            if init_type=="ReLU":
                print "ConvPoolLayer with He init"
                std = numpy.sqrt(2.0/fan_in)
                self.W = theano.shared(
                    numpy.asarray(
                        rng.normal(0, std, size=filter_shape),
                        dtype=theano.config.floatX
                    ),
                    borrow=True
                )
            else:
                print "ConvPoolLayer with Xavier init"
                fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                           numpy.prod(poolsize))
                # initialize weights with random weights
                W_bound = numpy.sqrt(6. / (fan_in + fan_out))    
                self.W = theano.shared(
                    numpy.asarray(
                        rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                        dtype=theano.config.floatX
                    ),
                    borrow=True
                )
        else:
            self.W = W

        if b==None:
            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b
            
        # convolve input feature maps with filters
        #conv_out = conv.conv2d(
        #    input=input,
        #    filters=self.W,
        #    filter_shape=filter_shape,
        #    image_shape=image_shape,
        #    border_mode='full'
        #)
        input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(stride=1, partial_sum=1, pad=pad)
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_filters = gpu_contiguous(filters_shuffled)
        conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

        # downsample each feature map individually, using maxpooling
        #pooled_out = downsample.max_pool_2d(
        #    input=conv_out,
        #    ds=poolsize,
        #    st=poolstride,
        #    ignore_border=False
        #)
        pool_op = MaxPool(ds=poolsize[0], stride=poolstride[0])
        pooled_out_shuffled = pool_op(conv_out_shuffled)
        pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
    
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        stride = 1# not used
        assert (image_shape[2]-filter_shape[2]+2*pad)%stride==0
        output_im_size = (image_shape[2]-filter_shape[2]+2*pad)/stride+1
        assert output_im_size%poolsize[0]==0
        output_im_size = output_im_size//poolsize[0]
        self.output_shape = [image_shape[0],
                            filter_shape[0],
                            output_im_size,
                            output_im_size]
                            
        # store parameters of this layer
        self.params = [self.W, self.b]

    def getOutputShape(self):
        return self.output_shape
    
    def load_params(self, W, b):
        self.W.set_value(W)
        self.b.set_value(b)
        print 'Convolutional Layer parameters loaded'
    def hasParams(self):
        return True
    def nParams(self):
        return 2

class MaxPoolingLayer():
    def __init__(self, input, image_shape, size=2, stride=2):
        assert (image_shape[2]-size)%stride==0
        self.output_shape = image_shape
        output_im_size = (image_shape[2]-size)/stride+1
        self.output_shape[2] = output_im_size
        self.output_shape[3] = output_im_size

        pool_op = MaxPool(ds=size, stride=stride)
        input_shuffled = input.dimshuffle(1,2,3,0)
        pool_out = pool_op(input_shuffled)
        self.output = pool_out.dimshuffle(3,0,1,2)
    def getOutputShape(self):
        return self.output_shape
    def hasParams(self):
        return False
    def nParams(self):
        return 0

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates

class ConvNet():
    def __init__(self, data_path, batch_size=50,
                 base_lr=0.04, momentum=0.0,
                 activation=T.tanh, params=[None]*100,
                 buf_size=1000, n_cand_chunk=50000, n_rot=0,
                 N_valid = 100000, N_test = 100000):
        
        self.data_path = data_path
        self.n_cand_chunk = n_cand_chunk
        
        self.batch_size = batch_size
        self.buf_size = buf_size

        self.N_valid = N_valid
        self.N_test = N_test

        # Validation params
        self.improvement_threshold = 0.99
        self.patience_increase = 2
        self.max_patience_increase = 100000
        
        rng = numpy.random.RandomState(23455)
        im_chan = 4

        self.im_chan = im_chan
        # Creation of validation and test sets
        self.train_set_x, self.train_set_y = shared_dataset((np.ones((1, 441*im_chan)), np.ones(1)))

        assert buf_size%batch_size==0
        assert buf_size>batch_size
        self.buf_train_set_x, self.buf_train_set_y = shared_dataset((np.ones((buf_size,441*im_chan)), np.ones(buf_size)))
        self.local_buf_x = self.buf_train_set_x.get_value()
        self.local_buf_y = self.buf_train_set_y.get_value()
        chunkLoader = ChunkLoader(data_path + '/chunks_validate/',
                                  n_cand_chunk, n_cand_chunk, n_rot = n_rot)

        v_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * im_chan))
        v_y = np.array([], dtype = "int32")
        while (len(v_y) < self.N_valid):
            v_x1, v_y1 = chunkLoader.getNext()
            v_x = np.vstack((v_x, v_x1))
            v_y = np.concatenate((v_y, v_y1))
        
        print "validation set = ", len(v_y)
        self.valid_set_x, self.valid_set_y = shared_dataset ([v_x, v_y])


        self.chunkLoader = ChunkLoader(data_path + '/chunks_train/',
                                  n_cand_chunk, batch_size, n_rot = n_rot)
    
        self.n_valid_batches = self.N_valid / batch_size
    
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.lr = T.fscalar() # learning rate symbolic variable
        #self.lr = theano.shared(np.asarray(base_lr,dtype=theano.config.floatX))
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
        
        im_size = 21
        self.input_data = self.x.reshape((batch_size, im_chan, im_size, im_size))
        print "input_data", (batch_size, im_chan, im_size, im_size)

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
        self.params = self.arch.getParams()
        self.cost = self.layers[-1].negative_log_likelihood(self.y)
        theano.printing.pydotprint(self.cost, outfile="./arch_graph.png", var_with_name_simple=True)
        self.learning_rate = base_lr
        self.train_model = theano.function(
            [self.index, self.lr],
            self.cost,
            updates=gradient_updates_momentum(self.cost, self.params, self.lr, momentum),
            givens={
                self.x: self.train_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: self.train_set_y[self.index * batch_size: (self.index + 1) * batch_size]
            }#, mode="DebugMode"
        )
        print 'train_model was compiled'
        theano.printing.pydotprint(self.train_model, outfile="./reduced_arch_graph.png", var_with_name_simple=True)
        self.validate_model = theano.function(
            [self.index],
            self.layers[-1].errors(self.y),
            givens={
                self.x: self.valid_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: self.valid_set_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        print 'validate_model was compiled'
        self.validation_loss = theano.function(
            [self.index],
            self.layers[-1].negative_log_likelihood(self.y),
            givens={
                self.x: self.valid_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: self.valid_set_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        print 'validation_loss was compiled'
        #self.validate_FPR = theano.function(
        #    [self.index],
        #    self.layer4.FPR(y),
        #    givens={
        #        self.x: self.valid_set_x[self.index * batch_size: (self.index + 1) * batch_size],
        #        self.y: self.valid_set_y[self.index * batch_size: (self.index + 1) * batch_size]
        #    },
        #    on_unused_input='warn'
        #)

        #self.validate_FNR = theano.function(
        #    [self.index],
        #    self.layer4.FNR(y),
        #    givens={
        #        self.x: self.valid_set_x[self.index * batch_size: (self.index + 1) * batch_size],
        #        self.y: self.valid_set_y[self.index * batch_size: (self.index + 1) * batch_size]
        #    },
        #    on_unused_input='warn'
        #)
    
        self.test_model_train = theano.function(
            [self.index],
            self.layers[-1].errors(self.y),
            givens={
                self.x: self.train_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: self.train_set_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        print 'test_model_train was compiled'
        self.train_buffer_error = theano.function(
            [self.index],
            self.layers[-1].errors(self.y),
            givens={
                self.x: self.buf_train_set_x[self.index * batch_size: (self.index + 1) * batch_size],
                self.y: self.buf_train_set_y[self.index * batch_size: (self.index + 1) * batch_size]
            }
        )
        print 'train_buffer_error was compiled'
        self.it = 0
        self.epoch = 0
        self.buf_index=0

        # training history (all iters)
        self.train_loss_history = []# loss
        self.train_err_history = []# error
        self.iter_train_history = []# iter

        # validation history
        self.val_err_history = []# error
        self.val_loss_history = []# loss
        self.iter_val_history = []# iter

        # buffer
        self.train_buf_err_history = []# error
        
        # best
        self.best_validation_error = numpy.inf
        self.patience_loss = numpy.inf
        self.best_iter = 0
        self.best_params = None
        
    def train(self):
        # load chunk data
        chunk_x, chunk_y = self.chunkLoader.getNext()
        self.train_set_x.set_value(chunk_x)
	self.train_set_y.set_value(chunk_y)

        # update buffer
        self.local_buf_x[self.buf_index:self.buf_index+self.batch_size] = chunk_x
        self.local_buf_y[self.buf_index:self.buf_index+self.batch_size] = chunk_y
        self.buf_index = (self.buf_index+self.batch_size)%self.buf_size
                
        
        # train loss
        DropoutLayer.activate()
        cost_train = self.train_model(0, self.learning_rate)

        # train error (minibatch)
        DropoutLayer.deactivate()
        train_minibatch_error = self.test_model_train(0)
        DropoutLayer.activate()

        # print loss every 100 iters
        if self.it%100==0:
            print "training @ iter =", self.it, ", loss = ", cost_train
            print "training @ iter =", self.it, ", error = ", train_minibatch_error

        # save loss and error
        self.train_loss_history.append(cost_train.tolist())
        self.train_err_history.append(train_minibatch_error)
        self.iter_train_history.append(self.it)

        # alert if train error is too high
        if train_minibatch_error > 0.1:
            print "--> train minibatch error = ", train_minibatch_error, " at iter ", self.it
            print "--> ", self.chunkLoader.current_file, self.chunkLoader.batch_i, self.chunkLoader.files[self.chunkLoader.current_file]

        self.it+=1

    def reduceLearningRate(self, gamma):
        self.learning_rate = self.learning_rate*gamma
        print "Learning rate: ", self.learning_rate

    def validate(self, patience):
        DropoutLayer.deactivate()
        print "validation @ iter", self.it

        # validation error
        sub_validation_errors = [self.validate_model(i) for i
                             in xrange(self.n_valid_batches)]
        validation_error = numpy.mean(sub_validation_errors)
	self.val_err_history.append(validation_error)
        
        # validation loss
        sub_validation_losses = [self.validation_loss(i) for i
                             in xrange(self.n_valid_batches)]
        validation_loss = numpy.mean(sub_validation_losses)
	self.val_loss_history.append(validation_loss)
        
        self.iter_val_history.append(self.it)

        # buffer
        self.buf_train_set_x.set_value(self.local_buf_x)
        self.buf_train_set_y.set_value(self.local_buf_y)
        sub_buffer_errors = [self.train_buffer_error(i) for i in xrange(self.buf_size/self.batch_size)]
        train_buf_err = numpy.mean(sub_buffer_errors)
        self.train_buf_err_history.append(train_buf_err)
        print('epoch %i, iter %i, train buffer error %f %%' %
              (self.epoch, self.it,
               train_buf_err * 100.))

        # print validation results
        print('epoch %i, iter %i, validation loss %f' %
              (self.epoch, self.it,
               validation_loss))
        print('epoch %i, iter %i, validation error %f %%' %
              (self.epoch, self.it,
               validation_error * 100.))
        print 'patience before checkBest', patience
        patience = self.checkBest(validation_error, patience)
        print 'patience after checkBest', patience
        DropoutLayer.activate()
        return patience

    def checkBest(self, validation_error, patience):
        if not validation_error < self.best_validation_error:
            return patience
        if validation_error < self.patience_loss * self.improvement_threshold:
            patience = max(patience, min((self.it * self.patience_increase,
                                          self.max_patience_increase + self.it)))
            print "Patience = ", patience
            self.patience_loss = validation_error

        # save best validation score and iteration number
        self.best_validation_error = validation_error
        self.best_iter = self.it
        self.best_params = [param.get_value() for param in self.params]
        return patience
        
    def endTraining(self):
        self.valid_set_x.set_value([[]])
        self.valid_set_y.set_value([])
        del(self.valid_set_x)
        del(self.valid_set_y)
        print "Free validation set" 
        
        self.train_set_x.set_value([[]])
        self.train_set_y.set_value([])
        del(self.train_set_x)
        del(self.train_set_y)
        print "Free training set"

    def load_params(self, params):
        self.arch.loadParams(params)
        
    def test(self):
        self.chunkLoader = ChunkLoader(self.data_path + '/chunks_test/',
                                       self.n_cand_chunk, self.n_cand_chunk, n_rot = 0)

        SNRs = []
        t_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * self.im_chan))
        t_y = np.array([], dtype = "int32")
        while (len(t_y) < self.N_test):
            t_x1, t_y1 = self.chunkLoader.getNext()
            t_x = np.vstack((t_x, t_x1))
            t_y = np.concatenate((t_y, t_y1))
            
            SNRs += self.chunkLoader.current_minibatch_SNR().tolist()
           
        print "test set = ", len(t_x)
        test_set_x, test_set_y = shared_dataset ([t_x, t_y])
        test_SNRs = np.array(SNRs)
        
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size

        print "compiling predict"
        predict = theano.function([self.index],self.layers[-1].p_y_given_x,
                                  givens={self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size]
                                  },
                                  on_unused_input='ignore')

        print "compiling test_model"
        # create a function to compute the mistakes that are made by the model
        test_model = theano.function(
            [self.index],
            self.layers[-1].errors(self.y),
            givens={
                self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            }
        )

        
        ############## TESTING #############
        print "Starting testing..."
        DropoutLayer.deactivate()
        
        self.load_params(self.best_params)
        test_pred = np.array([predict (i)
                              for i in xrange(n_test_batches)])
        test_pred = np.concatenate(test_pred, axis = 0)
        print 'test_pred:', test_pred 
        
        test_errors = np.array([test_model(i) for i in xrange(n_test_batches)])
        
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (self.best_validation_error * 100., self.best_iter, test_errors.mean() * 100.))

        ### Saving test results ###
        with open("test_predictions.pkl", "w") as f:
            pickle.dump({'ConvNet_pbbs': test_pred,
                         'labels': test_set_y.get_value(borrow=True),
                         'SNRs': test_SNRs},
                        f, pickle.HIGHEST_PROTOCOL)
    def save(self, name):
        header = name +"_"
        with open(header+"training_history.pkl", "w") as f:
            pickle.dump({'iter_train_history': self.iter_train_history,
                         'train_err_history': self.train_err_history,
                         'train_loss_history': self.train_loss_history},
                        f, pickle.HIGHEST_PROTOCOL)
        with open(header+"training_buffer_history.pkl", "w") as f:
            pickle.dump({'iter_train_buf_history': self.iter_val_history,
                         'train_buf_err_history': self.train_buf_err_history},
                        f, pickle.HIGHEST_PROTOCOL)
        with open(header+"validation_history.pkl", "w") as f:
            pickle.dump({'iter_val_history': self.iter_val_history,
                         'val_err_history': self.val_err_history,
                         'val_loss_history': self.val_loss_history},
                        f, pickle.HIGHEST_PROTOCOL)    
        with open(header+"convnet_state.pkl", "w") as f:
            pickle.dump({'params': [param.get_value() for param in self.params],
                         'best_params': self.best_params,
                         'best_iter': self.best_iter,
                         'best_validation_error': self.best_validation_error,
                         'patience_loss': self.patience_loss},
                        f, pickle.HIGHEST_PROTOCOL)    
        

def evaluate_convnet(data_path, n_cand_chunk, base_lr=0.04, stepsize=50000, gamma = 0.5, momentum=0.0,
                     n_epochs= 10000,
                     batch_size=50,
                     N_valid = 100000, N_test = 100000,
                     validate_every_batches = 2000, n_rot = 0, activation = T.tanh,
                     tiny_train = False, buf_size=1000, savestep=50000, resume=None):

    convnet = ConvNet(data_path, batch_size=batch_size,
                      base_lr=base_lr, momentum=momentum,
                      activation=activation,
                      buf_size=buf_size, n_cand_chunk=n_cand_chunk, n_rot=n_rot,
                      N_valid=N_valid, N_test=N_test)

    patience = 50000
    validation_frequency = min(validate_every_batches, patience/2)
    done_looping = False

    start_time = time.clock()
    epoch = 0

    if resume!=None:
        state = numpy.load(str(resume)+"_training_state.pkl")
        epoch = state['epoch']
        patience = state['patience']
        convnet.it = state['iteration']
        convnet.learning_rate = state['learning_rate']

        print "Loading training history"
        training = numpy.load(str(resume)+"_training_history.pkl")
        convnet.iter_train_history = training['iter_train_history']
        convnet.train_err_history = training['train_err_history']
        convnet.train_loss_history = training['train_loss_history']

        print "Loading validation history"
        validation = numpy.load(str(resume)+"_validation_history.pkl")
        convnet.iter_val_history = validation['iter_val_history']
        convnet.val_err_history = validation['val_err_history']
        convnet.val_loss_history = validation['val_loss_history']

        print "Loading buffer history"
        training_buffer = numpy.load(str(resume)+"_training_buffer_history.pkl")
        convnet.train_buf_err_history = training_buffer['train_buf_err_history']

        print "Loading convnet"
        convnet_state = np.load(str(resume)+"_convnet_state.pkl")
        convnet.load_params(convnet_state['params'])
        convnet.best_params = convnet_state['best_params']
        print "convnet.best_params loaded"
        convnet.best_iter = convnet_state['best_iter']
        print "convnet.best_iter set to ", convnet.best_iter
        convnet.best_validation_error = convnet_state['best_validation_error']
        print "convnet.best_validation_error set to ", convnet.best_validation_error
        convnet.patience_loss = convnet_state['patience_loss']
        print "convnet.patience_loss set to ", convnet.patience_loss
    DropoutLayer.activate()
    while epoch < n_epochs and (not done_looping):
        epoch_done = False
        print "Epoch ", epoch, ", iteration ", convnet.it,
        ", patience ", patience, ", learning rate ", convnet.learning_rate
        while not epoch_done:
            sys.stdout.flush()
            if convnet.it%savestep==0 and convnet.it>1:
                print "Saving @ iter ", convnet.it
                convnet.save(str(convnet.it))
                with open(str(convnet.it)+"_training_state.pkl", "w") as f:
                    pickle.dump({'epoch': epoch,
                                 'patience': patience,
                                 'iteration': convnet.it,
                                 'learning_rate': convnet.learning_rate},
                                f, pickle.HIGHEST_PROTOCOL)
        
            if convnet.it%stepsize==0 and convnet.it>1:
                convnet.reduceLearningRate(gamma)
            if convnet.it%validation_frequency==0 and convnet.it>1:
                patience = convnet.validate(patience)
            convnet.train()
            epoch_done = convnet.chunkLoader.done
            if patience <= convnet.it:
                done_looping = True
                print "patience <= iter", patience, convnet.it
                break
        convnet.chunkLoader.done = False
        epoch += 1
    elapsed_time = time.clock()-start_time
    print "Optimization complete"
    print >> sys.stderr, "Elapsed time: ", elapsed_time/60.0, " minutes" 
    convnet.test()
    convnet.save("final")

if __name__ == '__main__':
    print >> sys.stderr, "n_rot is hard coded to 0 for compatibility reasons. ChunkLoader might need to be refactored"
    c = ConfigParser ()
    c.read(sys.argv[1])
    
    if c.get("vars", "activation_function") == "tanh":
        activation = T.tanh
    elif c.get("vars", "activation_function") == "ReLU":
        activation = relu
    elif c.get("vars", "activation_function") == "PReLU":
        activation = prelu

    tiny_train = c.get("vars", "tiny_train")
    if tiny_train == "False":
        tiny_train = False
    else:
        tiny_train = int(tiny_train)

    resume = c.get("vars", "resume")
    if resume == "None":
        resume = None
    else:
        resume = int(resume)


    evaluate_convnet(c.get("vars", "path_to_chunks"),
                     int(c.get("vars", "n_cand_chunk")),
                     base_lr = float (c.get("vars", "base_lr")),
		     stepsize = int (c.get("vars", "stepsize")),
		     gamma = float (c.get("vars", "gamma")),
                     momentum = float (c.get("vars","momentum")),
                     n_epochs = int (c.get("vars", "n_epochs")),
                     batch_size = int (c.get("vars", "batch_size")),
                     N_valid = int (c.get("vars", "N_valid")),
                     N_test = int (c.get("vars", "N_test")),
                     #N_valid = 70000, N_test = 70000,
                     validate_every_batches = int (c.get("vars",
                                                         "validate_every_batches")),
                     n_rot = 0,
                     activation = activation,
    		     tiny_train = tiny_train,
                     buf_size=10000,
                     savestep =int (c.get("vars", "savestep")),
                     resume = resume
    )
