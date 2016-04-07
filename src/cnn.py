""" Architecture 7 / Model F with He initialization
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
from ConvNet import *
from DirectoryDataInterface import *

from ConfigParser import ConfigParser

def evaluate_convnet(arch_def, data_path, n_cand_chunk,
                     base_lr=0.04, stepsize=50000, gamma = 0.5, momentum=0.0,
                     n_epochs= 10000,
                     batch_size=50,
                     N_valid = 100000, N_test = 100000,
                     validate_every_batches = 2000, n_rot = 0, activation = T.tanh,
                     tiny_train = False, buf_size=1000, savestep=50000,
                     resume = None, improve_thresh = 0.99, ini_patience = 50000,
                     data_interface_str = "directory"):

    if data_interface_str == "directory":
        print "cnn.py: creating dataInterface"
        train_folder = data_path+'/chunks_train/'
        valid_folder = data_path+'/chunks_validate/'
        test_folder = data_path+'/chunks_test/'
        dataInterface = DirectoryDataInterface(train_folder, valid_folder, test_folder,
                                               n_cand_chunk = n_cand_chunk,
                                               batch_size = batch_size,
                                               N_valid = N_valid,
                                               N_test = N_test,
                                               N_train = N_train,
                                               im_chan = im_chan,
                                               im_size = im_size)
        print "cnn.py: dataInterface created"
    elif data_interface_str == "random":
        dataInterface = RandomDataInterface (data_path, 
                                               n_cand_chunk = n_cand_chunk,
                                               batch_size = batch_size,
                                               N_valid = N_valid,
                                               N_test = N_test,
                                               N_train = N_train,
                                               im_chan = im_chan,
                                               im_size = im_size)

    print "Creating ConvNet"
    convnet = ConvNet(dataInterface, arch_def, batch_size=batch_size,
                      base_lr=base_lr, momentum=momentum,
                      activation=activation,
                      buf_size=buf_size, n_cand_chunk=n_cand_chunk, n_rot=n_rot,
                      N_valid=N_valid, N_test=N_test)
    print "ConvNet created"
    
    patience = ini_patience
    validation_frequency = min(validate_every_batches, patience/2)
    done_looping = False

    start_time = time.clock()
    epoch = 0

    if resume != None:
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
            if convnet.it%savestep==0: # and convnet.it>1:
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
            if convnet.it%validation_frequency==0: # and convnet.it>1:
                patience = convnet.validate(patience)
            t = time.clock()
            convnet.train()
            print " training time = ", time.clock() - t
            epoch_done = convnet.dataInterface.doneTrain ()
            if patience <= convnet.it:
                done_looping = True
                print "patience <= iter", patience, convnet.it
                break
        #convnet.chunkLoader.done = False
        convnet.dataInterface.setDoneTrain (False)
        
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
        
    print c.get("vars", "arch_py")
    imp.load_source ("archpy", c.get("vars", "arch_py"))
    import archpy

    evaluate_convnet(archpy.convNetArchitecture(4, int (c.get("vars", "batch_size")),
                                                21, activation),
                     c.get("vars", "path_to_chunks"),
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
                     resume = resume,
                     improve_thresh = float (c.get("vars", "improvement_threshold")),
                     ini_patience = int (c.get("vars", "ini_patience")),
                     data_interface_str = c.get("vars", "data_interface")
    )
