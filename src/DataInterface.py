import numpy as np

class DataInterface ():
    # Return all validation data as numpy arrays
    def getValidationData (self):
        print "DataInterface:getValidationData not implemented"
        exit()

    # Return all training data as numpy arrays
    def getTrainingData (self):
        print "DataInterface:getTrainingData not implemented"
        exit()

    # Return all test data as numpy arrays
    def getTestData (self):
        print "DataInterface:getTestData not implemented"
        exit()

    # Return next minibatch in training set.
    def getNextTraining (self):
        print "DataInterface:getNextTraining not implemented"
        exit()

    # Return next minibatch in test set.
    def getNextTest (self):
        print "DataInterface:getNextTest not implemented"
        exit()

    # Return next minibatch in validation set.
    def getNextValidation (self):
        print "DataInterface:getNextValidation not implemented"
        exit()


    


