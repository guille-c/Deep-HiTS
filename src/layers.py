import numpy

import theano
import theano.tensor as T

#from theano.tensor.signal import downsample
#from theano.tensor.nnet import conv

#from theano.sandbox.cuda.basic_ops import gpu_contiguous
#from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
#from pylearn2.sandbox.cuda_convnet.pool import MaxPool

# Is being used. cuDNN 5
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.dnn import dnn_pool

from theano.sandbox.rng_mrg import MRG_RandomStreams
srng = MRG_RandomStreams()

def rectify(x, alpha=0):
    f1 = 0.5 * (1 + alpha)
    f2 = 0.5 * (1 - alpha)
    return f1 * x + f2 * abs(x)

def relu(x):
    return rectify(x, alpha=0)

def leaky_relu(x):
    return rectify(x, alpha=0.01)

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None, rng=None, init_type="tanh"):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
	if init_type=="ReLU":
	    print "Logistic Regression Layer with He init"
	    if rng==None:
		rng = numpy.random.RandomState(23455)
	    std = numpy.sqrt(2.0/n_in)
	    self.W = theano.shared(
		value=numpy.asarray(
		    rng.normal(0, std, size=(n_in, n_out)),
		    dtype=theano.config.floatX
		),
		name = 'W',
		borrow=True
	    )
	else:
	    print "Logistic Regression Layer with zero init"
	    # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
	    self.W = theano.shared(
		value=numpy.zeros(
		    (n_in, n_out),
		    dtype=theano.config.floatX
		),
		name='W',
		borrow=True
	    )
        if not W is None:
            self.W.set_value(W)

	# initialize the baises b as a vector of n_out 0s
	self.b = theano.shared(
	    value=numpy.zeros(
		(n_out,),
		dtype=theano.config.floatX
	    ),
	    name='b',
	    borrow=True
	)
        if not b is None:
            self.b.set_value(b)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
        self.output = self.p_y_given_x
        self.n_out = n_out
        self.batch_size = None
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
    def getOutputShape(self):
        return [self.batch_size, self.n_out]
    def hasParams(self):
        return True
    def nParams(self):
        return 2

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y],
                       dtype=theano.config.floatX)
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        ## check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y), dtype=theano.config.floatX)
        else:
            raise NotImplementedError()

    def FNR(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            #print T.mean(P)
            #TP = ((y == 1) & (self.y_pred == 1)).sum()
            zeros = T.zeros_like(y)
            ones = T.ones_like(y)
            N = T.eq(y, zeros)
            P = T.eq(y, ones)
            FN = T.and_(P, T.eq(zeros, self.y_pred))
            return T.mean(FN, dtype=theano.config.floatX)/T.mean(P,
                                                                 dtype=theano.config.floatX)
        else:
            raise NotImplementedError()

    def FPR(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            zeros = T.zeros_like(y)
            ones = T.ones_like(y)
            N = T.eq(y, zeros)
            P = T.eq(y, ones)
            FP = T.and_(N, T.eq(ones, self.y_pred))
            return T.mean(FP, dtype=theano.config.floatX)/T.mean(N,
                                                                 dtype=theano.config.floatX)
        else:
            raise NotImplementedError()
    def load_params(self, W, b):
        self.W.set_value(W)
        self.b.set_value(b)
        print "Logistic Regression parameters loaded"


class DropoutLayer(object):
    instances = []
    def __init__(self, input, p_drop=0.5):
        assert p_drop>=0 and p_drop<1

        self.input = input
        self.isActive = theano.shared(numpy.asarray(0,
                                                    dtype=theano.config.floatX),
                                      borrow=True)
        self.drop_prob = p_drop
        self.activated = srng.binomial(self.input.shape, p=(1-self.drop_prob), dtype='int32').astype('float32')
        self.output = self.isActive*self.activated*self.input +\
                      (1-self.isActive)*self.drop_prob*self.input
        self.instances.append(self)
        self.output_shape = None

    def setOutputShape(self, output_shape):
        self.output_shape = output_shape
    def getOutputShape(self):
        return self.output_shape
    def hasParams(self):
        return False
    def nParams(self):
        return 0
    @staticmethod
    def activate():
        for instance in DropoutLayer.instances:
            instance.isActive.set_value(1.0)

    @staticmethod
    def deactivate():
        for instance in DropoutLayer.instances:
            instance.isActive.set_value(0.0)

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh,init_type='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
	
	if init_type=="ReLU":
	    print "HiddenLayer with He init"
	    W_values = numpy.asarray(
		rng.normal(
		    0,
		    numpy.sqrt(2.0/n_in),
		    size=(n_in, n_out)
		),
                dtype=theano.config.floatX
	    )
	else:
	    print "HiddenLayer with Xavier init"
	    W_values = numpy.asarray(
		rng.uniform(
		    low=-numpy.sqrt(6. / (n_in + n_out)),
		    high=numpy.sqrt(6. / (n_in + n_out)),
		    size=(n_in, n_out)
		),
		dtype=theano.config.floatX
	    )
	if activation == theano.tensor.nnet.sigmoid:
	    W_values *= 4

	self.W = theano.shared(value=W_values, name='W', borrow=True)
	if not W is None:
	    self.W.set_value(W)

	b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
	self.b = theano.shared(value=b_values, name='b', borrow=True)
	if not b is None:
	    self.b.set_value(b)
	
        lin_output = T.dot(input, self.W) + self.b

        print 'hiddenlayer/lin_output dtypes (input, W, b, out)', input.dtype, self.W.dtype,
        print self.b.dtype, lin_output.dtype, 


        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        self.n_out = n_out
        self.batch_size = None
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
    def getOutputShape(self):
        return [self.batch_size, self.n_out]
                                      
    def load_params(self, W, b):
        self.W.set_value(W)
        self.b.set_value(b)
        print 'Hidden Layer parameters loaded'
    def hasParams(self):
        return True
    def nParams(self):
        return 2

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

#class ResidualLayer():
#    ### CHANGE WEIGHT INIT TO HE ET. AL. 2014 ###
#    def __init__(self, rng, input, filter_shape, image_shape,
#                 activation=leaky_relu, W1=None, W2=None, b1=None, b2=None):
#
#        assert image_shape[1] == filter_shape[1]
#        self.input = input
#
#	# there are "num input feature maps * filter height * filter width"
#	# inputs to each hidden unit
#	fan_in = numpy.prod(filter_shape[1:])
#	# each unit in the lower layer receives a gradient from:
#	# "num output feature maps * filter height * filter width" /
#	#   pooling size
#	fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
#	# initialize weights with random weights
#	W_bound = numpy.sqrt(6. / (fan_in + fan_out))
#	self.W1 = theano.shared(
#	    numpy.asarray(
#		rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
#		dtype=theano.config.floatX
#	    ),
#	    borrow=True
#	)
#        if not W1 is None:
#            self.W1.set_value(W1)
#
#	# the bias is a 1D tensor -- one bias per output feature map
#	b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
#	self.b1 = theano.shared(value=b_values, borrow=True)
#        if not b1 is None:
#            self.b1.set_value(b1)
#
#
#        assert filter_shape[2]%2==1# odd size
#        pad = (filter_shape[2]-1)//2
#            
#        input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        w1_shuffled = self.W1.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        conv_op1 = FilterActs(stride=1, partial_sum=1, pad=pad)
#        contiguous_input = gpu_contiguous(input_shuffled)
#        contiguous_w1 = gpu_contiguous(w1_shuffled)
#        conv_out_1_shuffled = conv_op1(contiguous_input, contiguous_w1)
#        conv_out_1 = conv_out_1_shuffled.dimshuffle(3, 0, 1, 2)# c01b to bc01
#        activ_1_out = activation(conv_out_1+self.b1.dimshuffle('x',0,'x','x'))
#
#
#        filter_shape[1] = filter_shape[0]
#
#	# there are "num input feature maps * filter height * filter width"
#	# inputs to each hidden unit
#	fan_in = numpy.prod(filter_shape[1:])
#	# each unit in the lower layer receives a gradient from:
#	# "num output feature maps * filter height * filter width" /
#	#   pooling size
#	fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
#	# initialize weights with random weights
#	W_bound = numpy.sqrt(6. / (fan_in + fan_out))
#	self.W2 = theano.shared(
#	    numpy.asarray(
#		rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
#		dtype=theano.config.floatX
#	    ),
#	    borrow=True
#	)
#        if not W2 is None:
#            self.W2.set_value(W2)
#
#	# the bias is a 1D tensor -- one bias per output feature map
#	b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
#	self.b2 = theano.shared(value=b_values, borrow=True)
#        if not b2 is None:
#            self.b2.set_value(b2)
#
#        w2_shuffled = self.W2.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        activ_1_out_shuffled = activ_1_out.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        contiguous_activ_1_out = gpu_contiguous(activ_1_out_shuffled)
#        contiguous_w2 = gpu_contiguous(w2_shuffled)
#        conv_op2 = FilterActs(stride=1, partial_sum=1, pad=pad)
#        conv_out_2_shuffled = conv_op2(contiguous_activ_1_out, contiguous_w2)
#        conv_out_2 = conv_out_2_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
#        self.output = activation(conv_out_2+self.b2.dimshuffle('x', 0, 'x', 'x')+input)
#
#        stride = 1# not used
#        assert (image_shape[2]-filter_shape[2]+2*pad)%stride==0
#        output_im_size = (image_shape[2]-filter_shape[2]+2*pad)/stride+1
#        self.output_shape = [image_shape[0],
#                            filter_shape[0],
#                            output_im_size,
#                            output_im_size]
#                            
#        # store parameters of this layer
#        self.params = [self.W1, self.b1, self.W2, self.b2]
#
#    def getOutputShape(self):
#        return self.output_shape
#    
#    def load_params(self, W1, b1, W2, b2):
#        self.W1.set_value(W1)
#        self.b1.set_value(b1)
#        self.W2.set_value(W2)
#        self.b2.set_value(b2)
#
#        print 'Residual Layer parameters loaded'
#    def hasParams(self):
#        return True
#    def nParams(self):
#        return 4
#
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,
                 pad = 0, poolsize=(2, 2), activation = T.tanh, poolstride=(2, 2),
                 init_type="tanh", conv_mode='conv',
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
        if not W is None:
            self.W.set_value(W)

	# the bias is a 1D tensor -- one bias per output feature map
	b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
	self.b = theano.shared(value=b_values, borrow=True)
        if not b is None:
            self.b.set_value(b)            

        # cuDNN implementation
        conv_out = dnn_conv(input, self.W,
                            border_mode=pad, conv_mode=conv_mode)
        pooled_out = dnn_pool(conv_out, poolsize,
                            stride=poolstride)
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
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
	
#class ConvPool2Layer(object):
#    """Pool Layer of a convolutional network """
#
#    def __init__(self, rng, input, filter_shape, image_shape,
#                 pad = 0, poolsize=(2, 2), activation = T.tanh, poolstride=(2, 2),
#                 init_type="tanh",
#                 W=None, b=None):
#        """
#        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
#
#        :type rng: numpy.random.RandomState
#        :param rng: a random number generator used to initialize weights
#
#        :type input: theano.tensor.dtensor4
#        :param input: symbolic image tensor, of shape image_shape
#
#        :type filter_shape: tuple or list of length 4
#        :param filter_shape: (number of filters, num input feature maps,
#                              filter height, filter width)
#
#        :type image_shape: tuple or list of length 4
#        :param image_shape: (batch size, num input feature maps,
#                             image height, image width)
#
#        :type poolsize: tuple or list of length 2
#        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
#        """
#
#        assert image_shape[1] == filter_shape[1]
#        self.input = input
#
#	# there are "num input feature maps * filter height * filter width"
#	# inputs to each hidden unit
#	fan_in = numpy.prod(filter_shape[1:])
#	# each unit in the lower layer receives a gradient from:
#	# "num output feature maps * filter height * filter width" /
#	#   pooling size
#	
#	if init_type=="ReLU":
#	    print "ConvPoolLayer with He init"
#	    std = numpy.sqrt(2.0/fan_in)
#	    self.W = theano.shared(
#		numpy.asarray(
#		    rng.normal(0, std, size=filter_shape),
#		    dtype=theano.config.floatX
#		),
#		borrow=True
#	    )
#	else:
#	    print "ConvPoolLayer with Xavier init"
#	    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
#		       numpy.prod(poolsize))
#	    # initialize weights with random weights
#	    W_bound = numpy.sqrt(6. / (fan_in + fan_out))    
#	    self.W = theano.shared(
#		numpy.asarray(
#		    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
#		    dtype=theano.config.floatX
#		),
#		borrow=True
#	    )
#        if not W is None:
#            self.W.set_value(W)
#
#	# the bias is a 1D tensor -- one bias per output feature map
#	b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
#	self.b = theano.shared(value=b_values, borrow=True)
#        if not b is None:
#            self.b.set_value(b)
#            
#        # convolve input feature maps with filters
#        #conv_out = conv.conv2d(
#        #    input=input,
#        #    filters=self.W,
#        #    filter_shape=filter_shape,
#        #    image_shape=image_shape,
#        #    border_mode='full'
#        #)
#        #input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        #filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
#        #conv_op = FilterActs(stride=1, partial_sum=1, pad=pad)
#        #contiguous_input = gpu_contiguous(input_shuffled)
#        #contiguous_filters = gpu_contiguous(filters_shuffled)
#        #conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
#	
#	conv_out = T.nnet.conv2d(input, self.W, border_mode=pad, filter_flip=False) 
#
#        # downsample each feature map individually, using maxpooling
#        pooled_out = downsample.max_pool_2d(
#            input=conv_out,
#            ds=poolsize,
#            st=poolstride,
#            ignore_border=False
#        )
#        #pool_op = MaxPool(ds=poolsize[0], stride=poolstride[0])
#        #pooled_out_shuffled = pool_op(conv_out_shuffled)
#        #pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01
#    
#	
#    
#        # add the bias term. Since the bias is a vector (1D array), we first
#        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
#        # thus be broadcasted across mini-batches and feature map
#        # width & height
#        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#        #self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
#
#        stride = 1# not used
#        assert (image_shape[2]-filter_shape[2]+2*pad)%stride==0
#        output_im_size = (image_shape[2]-filter_shape[2]+2*pad)/stride+1
#        assert output_im_size%poolsize[0]==0
#        output_im_size = output_im_size//poolsize[0]
#        self.output_shape = [image_shape[0],
#                            filter_shape[0],
#                            output_im_size,
#                            output_im_size]
#                            
#        # store parameters of this layer
#        self.params = [self.W, self.b]
#
#    def getOutputShape(self):
#        return self.output_shape
#    
#    def load_params(self, W, b):
#        self.W.set_value(W)
#        self.b.set_value(b)
#        print 'Convolutional Layer parameters loaded'
#    def hasParams(self):
#        return True
#    def nParams(self):
#        return 2


#class MaxPoolingLayer():
#    ## TODO: Use dnn_pool as pooling method
#    def __init__(self, input, image_shape, size=2, stride=2):
#        assert (image_shape[2]-size)%stride==0
#        self.output_shape = image_shape
#        output_im_size = (image_shape[2]-size)/stride+1
#        self.output_shape[2] = output_im_size
#        self.output_shape[3] = output_im_size
#
#        pool_op = MaxPool(ds=size, stride=stride)
#        input_shuffled = input.dimshuffle(1,2,3,0)
#        pool_out = pool_op(input_shuffled)
#        self.output = pool_out.dimshuffle(3,0,1,2)
#    def getOutputShape(self):
#        return self.output_shape
#    def hasParams(self):
#        return False
#    def nParams(self):
#        return 0
