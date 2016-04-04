import numpy as np

from DataInterface import *

class DirectoryDataInterface (DataInterface):

    def __init__ (self, trin_folder, valid_folder, test_folder):
        self.chunkLoaderTrain      #TBD
        self.chunkLoaderTest       #TBD
        self.chunkLoaderValidation #TBD
        
    # Return all validation data as numpy arrays
    def getValidationData (self):
        print "DirectoryDataInterface:getValidationData not implemented"
        exit()

    # Return all training data as numpy arrays
    def getTrainingData (self):
        print "DirectoryDataInterface:getTrainingData not implemented"
        exit()

    # Return all test data as numpy arrays
    def getTestData (self):
        print "DirectoryDataInterface:getTestData not implemented"
        exit()

    # Return next minibatch in training set.
    def getNextTraining (self):
        print "DirectoryDataInterface:getNextTraining not implemented"
        exit()

    # Return next minibatch in test set.
    def getNextTest (self):
        print "DirectoryDataInterface:getNextTest not implemented"
        exit()

    # Return next minibatch in validation set.
    def getNextValidation (self):
        print "DirectoryDataInterface:getNextValidation not implemented"
        exit()


    


