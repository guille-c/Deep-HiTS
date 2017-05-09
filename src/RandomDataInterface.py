import os
from ChunkDataInterface import *
from ListChunkLoader import *

class RandomDataInterface (ChunkDataInterface):

    def __init__ (self, folder,
                  n_cand_chunk = 50000, batch_size = 50,
                  N_valid = 100000, N_test = 100000, N_train = 1250000,
                  im_chan = 4, im_size = 21,
                  im_keys = ['temp_images', 'sci_images', 'diff_images', 'SNR_images']):
        self.folder = folder
        self.n_cand_chunk = n_cand_chunk
        self.batch_size = batch_size
        self.N_valid = N_valid
        self.N_test = N_test
        self.N_train = N_train
        self.im_chan = im_chan
        self.im_size = im_size
        self.keys = im_keys
        
	self.files = os.listdir(folder)
        self.files.sort()
        self.files = np.array(self.files)
        i_s = np.arange(len(self.files))
        np.random.shuffle (i_s)
        self.files = self.files[i_s]

        self.rows_per_file = {}
        # total_files = 0
        # print self.files
        # for f in self.files:
        #     print f
        #     chunk = np.load(folder + f)
        #     self.rows_per_file[f] = len(chunk["RF_pbbs"])
        #     total_files += 1
        #     if total_samples > N_valid + N_train + N_test:
        #         break
        # print "RandomDataInterface: total samples = ", total_samples
        # if N_valid + N_train + N_test > total_samples:
        #     print "RandomDataInterface: number of total samples is lower than " + \
        #         "required training, test, and validation datasets."
        #     exit()

        i_file = 0
        # Creating list of files for training
        self.chunkLoaderTrain, i_file = self.createChunkSet (N_train, i_file)

        # Creating list of files for validation
        self.chunkLoaderValidation, i_file = self.createChunkSet (N_valid, i_file)

        # Creating list of files for test
        self.chunkLoaderTest, i_file = self.createChunkSet (N_test, i_file)
        
    def createChunkSet (self, N, i_ini):
        cand_count = 0 # for counting candidates in this subset.
        files_set = []
        i_file = i_ini
        while (cand_count < N):
            chunk = np.load(self.folder + self.files[i_file])
            self.rows_per_file[self.files[i_file]] = len(chunk["RF_pbbs"])
            files_set.append(self.files[i_file])
            cand_count += self.rows_per_file[self.files[i_file]]
            i_file += 1
        print files_set
        print i_file, cand_count
        return ListChunkLoader (self.folder, files_set, self.n_cand_chunk,
                                self.batch_size, n_rot = 0, keys = self.keys), i_file


if __name__=="__main__":
    import sys
    rdi = RandomDataInterface (sys.argv[1], N_train = 40000, N_test = 15000, N_valid = 35000)


