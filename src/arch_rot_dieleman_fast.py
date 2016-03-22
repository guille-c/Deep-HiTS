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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer, DropoutLayer
from loadHITS import *
from ChunkLoader import *

from ConfigParser import ConfigParser

def relu(x):
    return T.switch(x<0, 0, x)
def prelu(x):
    return T.switch(x<0, 0.01*x, x)
    
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), activation = T.tanh, poolstride=(2, 2)):
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

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode='full'
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            st=poolstride,
            ignore_border=False
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        #self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output = activation(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.W, self.b]

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

def evaluate_convnet(data_path, n_cand_chunk, base_lr=0.1, stepsize=50000, gamma = 0.5, momentum=0.0,
                     n_epochs= 10000,
                     nkerns=[16, 32, 32, 512, 512], batch_size=50, isDropout=False,
                     N_valid = 100000, N_test = 100000,
                     validate_every_batches = 2000, n_rot = 0, activation = T.tanh,
                     tiny_train = False, buf_size=1000):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    print "nkerns = ", nkerns, nkerns.shape

    rng = numpy.random.RandomState(23455)
    #im_chan = 16
    im_chan = 4 * (n_rot + 1)

    # Creation of validation and test sets
    train_set_x, train_set_y = shared_dataset((np.ones((1, 441*im_chan)), np.ones(1)))

#    assert buf_size%batch_size==0
#    assert buf_size>batch_size
#    buf_train_set_x, buf_train_set_y = shared_dataset((np.ones((buf_size,441*im_chan)), np.ones(buf_size)))
#    local_buf_x = buf_train_set_x.get_value()
#    local_buf_y = buf_train_set_y.get_value()
    chunkLoader = ChunkLoader(data_path + '/chunks_validate/',
                              n_cand_chunk, n_cand_chunk, n_rot = n_rot)

    v_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * im_chan))
    v_y = np.array([], dtype = "int32")
    while (len(v_y) < N_test):
        v_x1, v_y1 = chunkLoader.getNext()
        v_x = np.vstack((v_x, v_x1))
        v_y = np.concatenate((v_y, v_y1))
        
    print "validation set = ", len(v_y)
    valid_set_x, valid_set_y = shared_dataset ([v_x, v_y])


    chunkLoader = ChunkLoader(data_path + '/chunks_train/',
                              n_cand_chunk, n_cand_chunk, n_rot = n_rot)
    
    if tiny_train:
        tr_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * im_chan))
        tr_y = np.array([], dtype = "int32")
        while (len(tr_y) < tiny_train):
            tr_x1, tr_y1 = chunkLoader.getNext()
            tr_x = np.vstack((tr_x, tr_x1))
            tr_y = np.concatenate((tr_y, tr_y1))
        
        print "training set = ", len(tr_y)
        train_set_x, train_set_y = shared_dataset ([tr_x, tr_y])

        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

    #valid_set_x, valid_set_y = datasets[1]
    #test_set_x, test_set_y   = datasets[2]
    

    #train_set_x = [theano.shared(np.zeros((1,1,1,1), dtype=theano.config.floatX)) for _ in xrange(N_chunk)]
    #train_set_y = theano.shared(np.zeros((1,1), dtype=theano.config.floatX))

    # compute number of minibatches for training, validation and testing
    #n_train_batches = 100000# train_set_x.get_value(borrow=True).shape[0]
    #n_valid_batches = #valid_set_x.get_value(borrow=True).shape[0]
    #n_train_batches /= batch_size
    #n_valid_batches /= batch_size
    n_valid_batches = N_valid / batch_size

    #print "n_train_batches = ", n_train_batches
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    lr = T.fscalar() # learning rate symbolic variable
    
    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 21 * 21)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (21, 21) is the size of MNIST images.
    im_size = 21
    input_data = x.reshape((batch_size, im_chan, im_size, im_size))
    print "input_data", (batch_size, im_chan, im_size, im_size)


    # Create 90 degrees rotations following Dieleman's strategy
    new_batch_size = batch_size*4
    part0 = input_data
    part1 = input_data[:,:,:,:-im_size-1:-1].dimshuffle(0,1,3,2) # 90 degrees
    part2 = input_data[:,:,:-im_size-1:-1,:-im_size-1:-1] # 180 degrees
    part3 = input_data[:,:,:-im_size-1:-1,:].dimshuffle(0,1,3,2) # 270 degrees

    layer0_input = T.concatenate([part0, part1, part2, part3], axis=0)

    # First ConvPool Layer
    filter_shape1 = 4 #8
    pool_size = 2
    pool_stride = 2
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        activation=activation,
        image_shape=(new_batch_size, im_chan, im_size, im_size),
        filter_shape=(nkerns[0], im_chan, filter_shape1, filter_shape1),
        poolsize=(pool_size, pool_size),
        poolstride=(pool_stride, pool_stride)
    )
    print "layer0 (filter)(pool)= ", (nkerns[0], im_chan, filter_shape1, filter_shape1), (pool_size, pool_size)
    maxpool_size1 = int(math.ceil(1.0*(im_size+filter_shape1 - 1)/pool_stride))
    print "output = ", maxpool_size1

    # Second ConvPool Layer (without pooling)
    filter_shape2 = 3 #6
    pool_size2 = 1
    pool_stride2 = 1
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        activation=activation,
        image_shape=(new_batch_size, nkerns[0], maxpool_size1, maxpool_size1),
        filter_shape=(nkerns[1], nkerns[0], filter_shape2, filter_shape2),
        poolsize=(pool_size2, pool_size2),
        poolstride=(pool_stride2, pool_stride2)
    )

    print "layer1 (filter)(pool) = ", (nkerns[1], nkerns[0], filter_shape2, filter_shape2), (pool_size2, pool_size2)
    maxpool_size2 = int(math.ceil(1.0*(maxpool_size1+filter_shape2 - 1)/pool_stride2))
    print "output= ", maxpool_size2

    # Third ConvPool Layer
    filter_shape3 = 3
    pool_size3 = 2
    pool_stride3 = 2
    layer2 = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        activation=activation,
        image_shape=(new_batch_size, nkerns[1], maxpool_size2, maxpool_size2),
        filter_shape=(nkerns[2], nkerns[1], filter_shape3, filter_shape3),
        poolsize=(pool_size3, pool_size3),
        poolstride=(pool_stride3, pool_stride3)
    )

    print "layer2 (filter)(pool) = ", (nkerns[2], nkerns[1], filter_shape3, filter_shape3), (pool_size3, pool_size3)

    maxpool_size3 = int(math.ceil(1.0*(maxpool_size2+filter_shape3 - 1)/pool_stride3))
    print "output= ", maxpool_size3
    
    # Recover structure (Dieleman's trick) 
    layer2_out_aux = layer2.output.reshape((4, batch_size, nkerns[2] * maxpool_size3 * maxpool_size3))
    layer3_input = layer2_out_aux.transpose(1,0,2).reshape((batch_size, 4*nkerns[2] * maxpool_size3 * maxpool_size3))

    # construct a fully-connected layer
    layer3 = HiddenLayer(
        rng,
        input=layer3_input,
        n_in=4*nkerns[2] * maxpool_size3 * maxpool_size3,
        n_out=nkerns[3],
        activation=activation
        #activation=T.tanh
        #activation=relu
    )
    print "Hidden units: ", nkerns[3]


    # #################DROPOUT##############
    if isDropout:
        print "Dropout ON"
        drop_layer3 = DropoutLayer(layer3.output, p_drop=0.5)    
        # classify the values of the fully-connected sigmoidal layer
        layer4 = LogisticRegression(input=drop_layer3.output, n_in=nkerns[3], n_out=2)
    else:
        print "Dropout OFF"
        layer4 = LogisticRegression(input=layer3.output, n_in=nkerns[3], n_out=2)
        
    # the cost we minimize during training is the NLL of the model
    cost = layer4.negative_log_likelihood(y)

    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # Shared Gradient Descent (SGD) Since this model has many
    # parameters, it would be tedious to manually create an update
    # rule for each model parameter. We thus create the updates list
    # by automatically looping over all (params[i], grads[i]) pairs.

    learning_rate = base_lr

    train_model = theano.function(
        [index, lr],
        cost,
        updates=gradient_updates_momentum(cost, params, lr, momentum),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }#, mode="DebugMode"
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validation_loss = theano.function(
        [index],
        layer4.negative_log_likelihood(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_FPR = theano.function(
        [index],
        layer4.FPR(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    )

    validate_FNR = theano.function(
        [index],
        layer4.FNR(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    )
    
    test_model_train = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
#    train_buffer_error = theano.function(
#        [index],
#        layer4.errors(y),
#        givens={
#            x: buf_train_set_x[index * batch_size: (index + 1) * batch_size],
#            y: buf_train_set_y[index * batch_size: (index + 1) * batch_size]
#        }
#    )
#    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    max_patience_increase = 100000
    improvement_threshold = 0.99  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(validate_every_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_error = numpy.inf
    patience_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    errors_val = []
    errors_test = []
    errors_train = []
    iters = []
    times = []
    #FPRs = []
    #FNRs = []
    train_err_history = []
    train_loss_history = []
    val_err_history = []
    val_loss_history = []
    iter_train_history = []
    iter_val_history = []
    best_params = params
    iter = 0
    #buf_index = 0
    #train_buf_err_history = []
    # Maximum number of epochs = n_epochs

    DropoutLayer.activate()
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches):
        epoch_done = False
        minibatch_index = 0
        while not epoch_done:
            chunk_x, chunk_y = chunkLoader.getNext()

            train_set_x.set_value(chunk_x)
	    train_set_y.set_value(chunk_y)        

            for mb_ind in range(n_cand_chunk//batch_size):
                iter += 1 #(epoch - 1) * n_train_batches + minibatch_index
                #print 'DEBUGGING', iter
                sys.stdout.flush()
                
                if tiny_train:
                    iter = (epoch - 1) * n_train_batches + minibatch_index
                    cost_ij = train_model(minibatch_index, learning_rate)
                    train_minibatch_error = test_model_train(minibatch_index)
                    minibatch_index += 1
                    epoch_done = (minibatch_index == n_train_batches)
                else:
                    
#                    local_buf_x[buf_index:buf_index+batch_size] = chunk_x
#                    local_buf_y[buf_index:buf_index+batch_size] = chunk_y
#                    
#                    buf_index = (buf_index+batch_size)%buf_size
#
                    #print "training mb_ind:", mb_ind, "iter", iter
                    cost_ij = train_model(mb_ind, learning_rate)
                    DropoutLayer.deactivate()
                    train_minibatch_error = test_model_train(0)
                    DropoutLayer.activate()
                    epoch_done = chunkLoader.done
                    
                if iter % 100 == 0:
                    print 'training @ iter = ', iter, ", cost = ", cost_ij
                train_loss_history.append(cost_ij.tolist())
                train_err_history.append(train_minibatch_error)
                iter_train_history.append(iter+1)
                
                if train_minibatch_error > 0.1:
                    print "--> train minibatch error = ", train_minibatch_error
                    print "--> current file", chunkLoader.current_file, " batch:", mb_ind, " filename", chunkLoader.files[chunkLoader.current_file]

	        # Adaptive Learning Rate
	        if (iter+1) % stepsize == 0:
                    learning_rate = learning_rate*gamma
		    #learning_rate.set_value(np.array(learning_rate.get_value()*gamma, dtype="float32"))
		    print "Learning rate: ", learning_rate#learning_rate.get_value()

	        #VALIDATION
                if (iter + 1) % validation_frequency == 0:
                    DropoutLayer.deactivate()
                    print "iter ", iter, " validation"
                    # compute zero-one loss on validation set
                    validation_errors = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_error = numpy.mean(validation_errors)
		    val_err_history.append(this_validation_error)

                    validation_losses = [validation_loss(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
		    val_loss_history.append(this_validation_loss)
                    
                    iter_val_history.append(iter+1)
                    print('epoch %i, iter %i, validation loss %f' %
                          (epoch, iter + 1,
                           this_validation_loss))
                    
                    print('epoch %i, iter %i, validation error %f %%' %
                          (epoch, iter + 1,
                           this_validation_error * 100.))
                    print('epoch %i, iter %i, train minibatch error %f %%' %
                          (epoch, iter + 1,
                           train_minibatch_error * 100.))
                
                    # Error calculation for the training buffer
                    if not tiny_train:
#                        buf_train_set_x.set_value(local_buf_x)
#                        buf_train_set_y.set_value(local_buf_y)
#                        buffer_errors = [train_buffer_error(i) for i in xrange(buf_size/batch_size)]
#                        train_buf_err = numpy.mean(buffer_errors)
#                        train_buf_err_history.append(train_buf_err)
#                        print('epoch %i, iter %i, train buffer error %f %%' %
#                              (epoch, iter + 1,
#                               train_buf_err * 100.))
#                        # if we got the best validation score until now
                        if this_validation_error < best_validation_error:
                            
                            #improve patience if loss improvement is good enough
                            if this_validation_error < patience_loss *  \
                               improvement_threshold:
                                patience = max(patience, min((iter * patience_increase, max_patience_increase + iter)))
                                print "patience = ", patience, improvement_threshold, iter * patience_increase
                                patience_loss = this_validation_error
                                # save best validation score and iteration number
                            best_validation_error = this_validation_error
                            best_iter = iter
                            best_params = params

                        ## test it on the test set
                        #test_losses = [
                        #    test_model(i)
                        #    for i in xrange(n_test_batches)
                        #]
                        #test_score = numpy.mean(test_losses)


                        # test it on the train set
                        #train_losses = [
                        #    test_model_train(i)
                        #    for i in xrange(n_test_batches)
                        #]
                        #train_score = numpy.mean(train_losses)
                        
                        #val_FPR = [validate_FPR(i) for i
                        #                 in xrange(n_valid_batches)]
                        #FPR = numpy.mean(val_FPR)
                        #val_FNR = [validate_FNR(i) for i
                        #                 in xrange(n_valid_batches)]
                        #FNR = numpy.mean(val_FNR)

                        #print(('     epoch %i, minibatch %i/%i, test error of '
                        #       'best model %f %%') %
                        #      (epoch, minibatch_index + 1, n_train_batches,
                        #       test_score * 100.))

                        #print "FPR, FNR = ", FPR, FNR
                    
                        errors_val.append(best_validation_error * 100)
                        #errors_test.append(test_score * 100)
                        #errors_train.append(train_score * 100)
                        iters.append(iter)
                        times.append(time.clock()-start_time)
                        #FPRs.append(FPR)_
                        #FNRs.append(FNR)
                    DropoutLayer.activate()

                if patience <= iter:
                    done_looping = True
                    print "patience <= iter", patience, iter
                    break

        chunkLoader.done = False

    end_time = time.clock()
    print('Optimization complete.')
    
    #np.save("ConvNets_HITS",
    #        np.array([iters, errors_train, errors_val, errors_test,
    #                  FPRs, FNRs, times]))


    # validate_model = theano.function(
    #     [index],
    #     layer3.errors(y),
    #     givens={
    #         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
    #         y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    #     }
    # )
    print "batch_size = ", batch_size
    valid_set_x.set_value([[]])
    valid_set_y.set_value([])
    del(valid_set_x)
    del(valid_set_y)

    train_set_x.set_value([[]])
    train_set_y.set_value([])
    del(train_set_x)
    del(train_set_y)

    # Loading test data
    chunkLoader = ChunkLoader(data_path + '/chunks_test/',
                              n_cand_chunk, n_cand_chunk, n_rot = n_rot)

    SNRs = []
    t_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * im_chan))
    t_y = np.array([], dtype = "int32")
    while (len(t_y) < N_test):
        t_x1, t_y1 = chunkLoader.getNext()
        t_x = np.vstack((t_x, t_x1))
        t_y = np.concatenate((t_y, t_y1))
        
        SNRs += chunkLoader.current_minibatch_SNR().tolist()
        #print 'tamanos test set (snr, x, y):', len(SNRs), len(t_x), len(t_y)
    print "test set = ", len(t_x)
    test_set_x, test_set_y = shared_dataset ([t_x, t_y])
    test_SNRs = np.array(SNRs)
    #print 'test_SNRs', test_SNRs
    
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches /= batch_size
    
    predict = theano.function([index],layer4.p_y_given_x,
                              givens={x:test_set_x[index * batch_size: (index + 1) * batch_size]},
                              on_unused_input='ignore')

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ############## TESTING #############
    DropoutLayer.deactivate()
    
    params = best_params
    test_pred = np.array([predict (i)
                          for i in xrange(n_test_batches)])
    test_pred = np.concatenate(test_pred, axis = 0)
    print 'test_pred:', test_pred
    #test_pred = []
    #for i in xrange (n_test_batches):
    #    test_pred += predict(i) 

    test_errors = np.array([test_model(i) for i in xrange(n_test_batches)])
    #print 'test_errors', test_errors, test_errors.mean()
    #print test_pred.shape, test_pred[0].shape
    #    test_pred = predict()

    #print "test_pred = ", len(test_pred)
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_error * 100., best_iter + 1, test_errors.mean() * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    #pkl_out = np.load(sys.argv[2])
    #with open ("test_predictions.pkl", "w") as f:
    #    pickle.dump({'ConvNet_pbbs': test_pred,
    #                 'labels': pkl_out['labels'],
    #                 'SNRs': pkl_out['SNRs']},
    #                f, pickle.HIGHEST_PROTOCOL)
    with open("training_history.pkl", "w") as f:
        pickle.dump({'iter_train_history': iter_train_history,
                     'train_err_history': train_err_history,
                     'train_loss_history': train_loss_history},
                    f, pickle.HIGHEST_PROTOCOL)
    #with open("training_buffer_history.pkl", "w") as f:
    #    pickle.dump({'iter_train_buf_history': iter_val_history,
    #                 'train_buf_err_history': train_buf_err_history},
    #                f, pickle.HIGHEST_PROTOCOL)
    with open("validation_history.pkl", "w") as f:
        pickle.dump({'iter_val_history': iter_val_history,
                     'val_err_history': val_err_history,
                     'val_loss_history': val_loss_history},
                    f, pickle.HIGHEST_PROTOCOL)

    with open("test_predictions.pkl", "w") as f:
        pickle.dump({'ConvNet_pbbs': test_pred,
                     'labels': test_set_y.get_value(borrow=True),
                     'SNRs': test_SNRs},
                    f, pickle.HIGHEST_PROTOCOL)
    
    np.save ("parameters", params)
    #with open("learning_history.pkl", "w") as f:
    #    pickle.dump({'iter_history': iter_history, 'train_err_history': 
    #                 train_err_history, 'val_err_history': val_err_history, 
    #                 'test_err':test_errors.mean()}, f,  pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    c = ConfigParser ()
    c.read(sys.argv[1])
    #print type(c.get("vars", "a"))
    #print c.get("vars", "c")

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

    evaluate_convnet(c.get("vars", "path_to_chunks"),
                     int(c.get("vars", "n_cand_chunk")),
                     base_lr = float (c.get("vars", "base_lr")),
		     stepsize = int (c.get("vars", "stepsize")),
		     gamma = float (c.get("vars", "gamma")),
                     momentum = float (c.get("vars","momentum")),
                     n_epochs = int (c.get("vars", "n_epochs")),
                     #nkerns=[20, 50],
                     nkerns = np.array(c.get("vars", "nkerns").split (","), dtype = int),
                     batch_size = int (c.get("vars", "batch_size")),
                     N_valid = int (c.get("vars", "N_valid")),
                     N_test = int (c.get("vars", "N_test")),
                     #N_valid = 70000, N_test = 70000,
                     validate_every_batches = int (c.get("vars",
                                                         "validate_every_batches")),
                     n_rot = int (c.get("vars", "n_rot")),
                     isDropout = (c.get("vars", "isDropout")=="True"),
                     activation = activation,
    		     tiny_train = tiny_train)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
