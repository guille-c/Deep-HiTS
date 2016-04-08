import numpy as np

from DataInterface import *
from ChunkLoader import *

class ChunkDataInterface (DataInterface):

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

