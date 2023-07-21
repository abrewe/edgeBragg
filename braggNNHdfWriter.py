import time
import numpy as np
import h5py
from pvapy.hpc.userMpDataProcessor import UserMpDataProcessor

class BraggNNHdfWriter(UserMpDataProcessor):

    def __init__(self, writerId, fileName, compression):
        UserMpDataProcessor.__init__(self)
        self.writerId = writerId
        self.fileName = fileName
        self.compression = compression
        self.logger.debug(f'Using file {fileName} for writer {writerId}, compression is {compression}')
        self.h5fd = None

        self.resetStats()

    def process(self, mpqObject):
        t0 = time.time()
        ddict = mpqObject
        if self.h5fd is None:
            self.h5fd = h5py.File(self.fileName, 'w')
            for key, data in ddict.items():
                if type(data) != np.ndarray:
                    continue
                dshape = list(data.shape)
                dshape[0] = None
                if self.compression:
                    self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape, compression="gzip")
                else:
                    self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape)
                self.logger.debug(f'Created dataset {key} with {data.shape} samples')
        else:
            for key, data in ddict.items():
                if type(data) != np.ndarray:
                    continue
                self.h5fd[key].resize((self.h5fd[key].shape[0] + data.shape[0]), axis=0)
                self.h5fd[key][-data.shape[0]:] = data
                self.logger.debug(f'Added {data.shape} samples to key {key}')
        self.h5fd.flush()
        writeTime = time.time() - t0
        self.writeTimeSum += writeTime
        self.logger.debug(f'File {self.fileName} written in {writeTime:.4f} seconds')
        self.nWritten += 1

    def getStats(self):
        writeTime = 0.0
        if self.nWritten > 0:
            writeTime = self.writeTimeSum/self.nWritten
        statsDict = {
            'nObjectsWritten' : self.nWritten,
            'writeTime' : writeTime
        }
        return statsDict

    def resetStats(self):
        self.nWritten = 0
        self.writeTimeSum = 0.0

