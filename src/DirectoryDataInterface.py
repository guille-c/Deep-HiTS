import numpy as np

from DataInterface import *
from ChunkLoader import *

class DirectoryDataInterface (DataInterface):

    def __init__ (self, train_folder, valid_folder, test_folder,
                  n_cand_chunk = 50000, batch_size = 50,
                  N_valid = 100000, N_test = 100000, N_train = 1250000,
                  im_chan = 4, im_size = 21):
        self.n_cand_chunk = n_cand_chunk
        self.batch_size = batch_size
        self.N_valid = N_valid
        self.N_test = N_test
        self.N_train = N_train
        self.im_chan = im_chan
        self.im_size = im_size

        print "DirectoryDataInterface: creating chunkLoaderTrain"
        self.chunkLoaderTrain = ChunkLoader(train_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)
        print "DirectoryDataInterface: chunkLoaderTrain created"
        print "DirectoryDataInterface: creating chunkLoaderValidation"
        self.chunkLoaderValidation = ChunkLoader(valid_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)
        print "DirectoryDataInterface: chunkLoaderValidation created"
        print "DirectoryDataInterface: creating chunkLoaderTest"                
        self.chunkLoaderTest = ChunkLoader(test_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)
        print "DirectoryDataInterface: chunkLoaderTest created"

    def getDataFromChunkLoader(self, chunkLoader, N_data, get_SNRs = False):
        # REMEMBER CASTING DATA TO theano.config.floatX and int32
        v_x = np.array([]).reshape((0, self.im_size**2 * self.im_chan))
        v_y = np.array([])
        if get_SNRs:
            SNRs = []
        while (len(v_y) < N_data):
            v_x1, v_y1 = chunkLoader.getNext()
            v_x = np.vstack((v_x, v_x1))
            v_y = np.concatenate((v_y, v_y1))
            if get_SNRs:
                SNRs += chunkLoader.current_minibatch_SNR().tolist()
        if get_SNRs:
            return (v_x, v_y, SNRs)
        else:
            return (v_x, v_y)
        
    # Return all validation data as numpy arrays
    def getValidationData (self):
        return self.getDataFromChunkLoader(self.chunkLoaderValidation,
                                           self.N_valid)

    # Return all training data as numpy arrays
    def getTrainingData (self):
        return self.getDataFromChunkLoader(self.chunkLoaderTrain,
                                           self.N_train)
        
    # Return all test data as numpy arrays
    def getTestData (self, get_SNRs = False):
        return self.getDataFromChunkLoader(self.chunkLoaderTest,
                                           self.N_test, get_SNRs = get_SNRs)

    # Return next minibatch in training set.
    def getNextTraining (self):
        return self.chunkLoaderTrain.getNext()
    
    # Return next minibatch in test set.
    def getNextTest (self):
        return self.chunkLoaderTest.getNext()
    
    # Return next minibatch in validation set.
    def getNextValidation (self):
        return self.chunkLoaderValidation.getNext()
    
    def doneTrain(self):
        return self.chunkLoaderTrain.done

    def setDoneTrain (self, done):
        self.chunkLoaderTrain.done = done

