import os
import sys
import time

from ChunkLoader import *

class ListChunkLoader(ChunkLoader):
    def __init__(self, folder, files, n_cand_chunk, batch_size, n_rot = 3,
                 keys = ['temp_images', 'sci_images', 'diff_images', 'SNR_images']):
	self.files = files
	self.current_file = 0
	self.batch_i = 0
        self.folder = folder
        self.n_cand_chunk = n_cand_chunk
        self.batch_size = batch_size
	self.current_file_data = np.load(self.folder+self.files[self.current_file])
	self.lastSNRs = []
        self.done = False
        self.n_rot = n_rot
        self.keys = keys
