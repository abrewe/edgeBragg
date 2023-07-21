import time
import zmq
from pvapy.hpc.userMpDataProcessor import UserMpDataProcessor

class BraggNNZmqWriter(UserMpDataProcessor):

    def __init__(self, port):
        UserMpDataProcessor.__init__(self)
        self.port = port
        self.context = None
        self.publisher = None

        self.resetStats()

    def process(self, mpqObject):
        if not self.context:
            self.context = zmq.Context()
            self.publisher = self.context.socket(zmq.PUB)
            self.publisher.bind(f'tcp://*:{self.port}')
        t0 = time.time()
        returnCode = self.publisher.send_pyobj(mpqObject)
        publishTime = time.time() - t0
        if returnCode != 0:
            self.publishTimeSum += publishTime
            self.nPublished += 1
            self.logger.debug(f'Datasets {mpqObject.keys()} have been published via ZMQ in {publishTime:.6f} seconds')
        else:
            self.nErrors += 1
            self.logger.error(f'Failed to publish {mpqObject.keys()} via ZMQ, return code: {returnCode}')

    def getStats(self):
        publishTime = 0.0
        if self.nPublished > 0:
            publishTime = self.publishTimeSum/self.nPublished
        statsDict = {
            'nObjectsPublished' : self.nPublished,
            'nErrors' : self.nErrors,
            'publishTime' : publishTime
        }
        return statsDict

    def resetStats(self):
        self.nPublished = 0
        self.publishTimeSum = 0.0
        self.nErrors = 0
    
