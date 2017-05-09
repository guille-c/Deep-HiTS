import theano.tensor as T
import numpy as np
from ArchBuilder import *
from time import time

from loadHITS import *
from ChunkLoader import *

class ConvNet():
    def __init__(self, data_interface, arch_def,
                 batch_size=50,
                 base_lr=0.04, momentum=0.0,
                 activation=T.tanh, params=[None]*100,
                 buf_size=1000, n_cand_chunk=50000, n_rot=0,
                 N_valid = 100000, N_test = 100000, im_chan = 4,
                 im_size = 21, improvement_threshold = 0.99):

        # self.data_path = data_path
        self.n_cand_chunk = n_cand_chunk

        self.batch_size = batch_size
        self.buf_size = buf_size

        self.N_valid = N_valid
        self.N_test = N_test

        N_train = 1250000
        # train_folder = data_path+'/chunks_train/'
        # valid_folder = data_path+'/chunks_validate/'
        # test_folder = data_path+'/chunks_test/'

        self.dataInterface = data_interface
        # print "ConvNet: creating dataInterface"
        # self.dataInterface = DirectoryDataInterface(train_folder, valid_folder, test_folder,
        #                                             n_cand_chunk = n_cand_chunk,
        #                                             batch_size = batch_size,
        #                                             N_valid = N_valid,
        #                                             N_test = N_test,
        #                                             N_train = N_train,
        #                                             im_chan = im_chan,
        #                                             im_size = im_size)
        # print "ConvNet: dataInterface created"
        
        # Validation params
        self.improvement_threshold = improvement_threshold
        self.patience_increase = 2
        self.max_patience_increase = 100000
        
        #rng = np.random.RandomState(23455)
        rng = np.random.RandomState()

        self.im_chan = im_chan
        # Creation of validation and test sets
        self.train_set_x, self.train_set_y = shared_dataset((np.ones((1, 441*im_chan)), np.ones(1)))

        assert buf_size%batch_size==0
        assert buf_size>batch_size
        self.buf_train_set_x, self.buf_train_set_y = shared_dataset((np.ones((buf_size,441*im_chan)), np.ones(buf_size)))
        self.local_buf_x = self.buf_train_set_x.get_value()
        self.local_buf_y = self.buf_train_set_y.get_value()
        # chunkLoader = ChunkLoader(data_path + '/chunks_validate/',
        #                           n_cand_chunk, n_cand_chunk, n_rot = n_rot)
        # 
        # v_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * im_chan))
        # v_y = np.array([], dtype = "int32")
        # while (len(v_y) < self.N_valid):
        #     v_x1, v_y1 = chunkLoader.getNext()
        #     v_x = np.vstack((v_x, v_x1))
        #     v_y = np.concatenate((v_y, v_y1))

        print "ConvNet: getting validation data"
        v_x, v_y = self.dataInterface.getValidationData ()
        v_x, v_y = v_x.astype(th.config.floatX), v_y.astype("int32")
        print "validation set = ", len(v_y)
        self.valid_set_x, self.valid_set_y = shared_dataset ([v_x, v_y])


        # self.chunkLoader = ChunkLoader(data_path + '/chunks_train/',
        #                           n_cand_chunk, batch_size, n_rot = n_rot)
    
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
        
        self.input_data = self.x.reshape((batch_size, im_chan, im_size, im_size))
        print "input_data", (batch_size, im_chan, im_size, im_size)

        self.arch_def = arch_def
                     
        self.arch = ArchBuilder(self.arch_def, self.input_data, params, rng)
        self.layers = self.arch.getLayers()
        self.params = self.arch.getParams()
        self.cost = self.layers[-1].negative_log_likelihood(self.y)
        #theano.printing.pydotprint(self.cost, outfile="./arch_graph.png", var_with_name_simple=True)
        self.learning_rate = base_lr
        
        self.train_model = theano.function(
            [self.index, self.lr],
            self.cost,
            updates=gradient_updates_momentum(self.cost, self.params, self.lr, momentum),
            givens={
                self.y: self.train_set_y[self.index * batch_size: (self.index + 1) * batch_size],
                self.x: self.train_set_x[self.index * batch_size: (self.index + 1) * batch_size]
            }#, mode="DebugMode"
        )
        print 'train_model was compiled'

        ### Draw optimized model ###
        #theano.printing.pydotprint(self.train_model, outfile="./reduced_arch_graph.png", var_with_name_simple=True)
        ### It takes about 1 minute to finish
        
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
        self.best_validation_error = np.inf
        self.patience_loss = np.inf
        self.best_iter = 0
        self.best_params = None
        
    def train(self):
        # load chunk data
        chunk_x, chunk_y = self.dataInterface.getNextTraining ()
        self.train_set_x.set_value(chunk_x.astype(th.config.floatX))
	self.train_set_y.set_value(chunk_y.astype("int32"))

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

        # # alert if train error is too high
        # if train_minibatch_error > 0.1:
        #     print "--> train minibatch error = ", train_minibatch_error, " at iter ", self.it
        #     print "--> ", self.dataInterface.chunkLoaderTrain.current_file, \
        #         self.dataInterface.chunkLoaderTrain.batch_i, \
        #         self.dataInterface.chunkLoaderTrain.files[self.dataInterface.chunkLoaderTrain.current_file]

        self.it+=1

    def reduceLearningRate(self, gamma):
        self.learning_rate = self.learning_rate*gamma
        print "Learning rate: ", self.learning_rate

    def validate(self, patience):
        t = time.time()
        DropoutLayer.deactivate()
        print "validation @ iter", self.it

        # validation error
        sub_validation_errors = [self.validate_model(i) for i
                             in xrange(self.n_valid_batches)]
        validation_error = np.mean(sub_validation_errors)
	self.val_err_history.append(validation_error)
        
        # validation loss
        sub_validation_losses = [self.validation_loss(i) for i
                             in xrange(self.n_valid_batches)]
        validation_loss = np.mean(sub_validation_losses)
	self.val_loss_history.append(validation_loss)
        
        self.iter_val_history.append(self.it)

        # buffer
        self.buf_train_set_x.set_value(self.local_buf_x)
        self.buf_train_set_y.set_value(self.local_buf_y)
        sub_buffer_errors = [self.train_buffer_error(i) for i in xrange(self.buf_size/self.batch_size)]
        train_buf_err = np.mean(sub_buffer_errors)
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
        print " validation time = ", time.time() - t
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
        # self.chunkLoader = ChunkLoader(self.data_path + '/chunks_test/',
        #                                self.n_cand_chunk, self.n_cand_chunk, n_rot = 0)
        # 
        # SNRs = []
        # t_x = np.array([], dtype = th.config.floatX).reshape((0, 441 * self.im_chan))
        # t_y = np.array([], dtype = "int32")
        # while (len(t_y) < self.N_test):
        #     t_x1, t_y1 = self.chunkLoader.getNext()
        #     t_x = np.vstack((t_x, t_x1))
        #     t_y = np.concatenate((t_y, t_y1))
        #     
        #     SNRs += self.chunkLoader.current_minibatch_SNR().tolist()

        t_x, t_y, SNRs = self.dataInterface.getTestData (get_SNRs = True)
        t_x, t_y = t_x.astype(th.config.floatX), t_y.astype("int32")

        print "test set = ", len(t_x)
        
        test_set_x, test_set_y = shared_dataset ([t_x, t_y])
        test_SNRs = np.array(SNRs)
        
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size

        print "compiling predict"
        predict = theano.function([self.index],self.layers[-1].p_y_given_x,
                                  givens={self.x: test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size]
                                  },
                                  on_unused_input='ignore')# <- Theano error with float64
        
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
        updates.append((param_update, momentum*param_update + learning_rate*T.grad(cost, param)))
        updates.append((param, param - param_update))
    return updates        
