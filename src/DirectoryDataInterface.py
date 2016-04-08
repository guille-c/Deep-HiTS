import numpy as np

from ChunkDataInterface import *
from ChunkLoader import *

class DirectoryDataInterface (ChunkDataInterface):

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

