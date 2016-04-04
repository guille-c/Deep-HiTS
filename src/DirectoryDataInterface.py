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
        self.chunkLoaderTrain = ChunkLoader(train_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)
        self.chunkLoaderValidation = ChunkLoader(valid_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)
        self.chunkLoaderTest = ChunkLoader(test_folder, self.n_cand_chunk,
                                            self.batch_size, n_rot = 0)

    def getDataFromChunkLoader(self, chunkLoader, N_data):
        # REMEMBER CASTING DATA TO theano.config.floatX and int32
        v_x = np.array([]).reshape((0, self.im_size**2 * self.im_chan))
        v_y = np.array([])
        while (len(v_y) < self.N_data):
            v_x1, v_y1 = chunkLoader.getNext()
            v_x = np.vstack((v_x, v_x1))
            v_y = np.concatenate((v_y, v_y1))
        return (v_x, v_y)
        
    # Return all validation data as numpy arrays
    def getValidationData (self):
        return getDataFromChunkLoader(self.chunkLoaderValidation, N_valid)

    # Return all training data as numpy arrays
    def getTrainingData (self):
        return getDataFromChunkLoader(self.chunkLoaderTrain, N_train)
        
    # Return all test data as numpy arrays
    def getTestData (self):
        return getDataFromChunkLoader(self.chunkLoaderTest, N_test)

    # Return next minibatch in training set.
    def getNextTraining (self):
        return self.chunkLoaderTrain.getNext()
    
    # Return next minibatch in test set.
    def getNextTest (self):
        return self.chunkLoaderTest.getNext()
    
    # Return next minibatch in validation set.
    def getNextValidation (self):
        return self.chunkLoaderValidation.getNext()

    


