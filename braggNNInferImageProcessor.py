import os
import numpy as np
import multiprocessing as mp
import threading
import queue
import time
import yaml
import pvapy as pva
from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.hpc.userMpWorkerController import UserMpWorkerController 
from braggNNFrameProcessor import BraggNNFrameProcessor
from braggNNHdfWriter import BraggNNHdfWriter
from braggNNZmqWriter import BraggNNZmqWriter

class BraggNNInferImageProcessor(AdImageProcessor):

    Q_WAIT_TIME = 1

    FRAME_PROCESSOR_WORKER_ID = 'frameProcessor'
    FRAME_HDF_WRITER_WORKER_ID = 'frameHdfWriter'
    PEAK_HDF_WRITER_WORKER_ID = 'peakHdfWriter'
    PEAK_ZMQ_WRITER_WORKER_ID = 'peakZmqWriter'

    def __init__(self, configDict={}):
        AdImageProcessor.__init__(self, configDict)
        self.configFile = configDict.get('configFile')
        if not self.configFile:
            raise Exception('No configuration file provided')
        self.params = yaml.load(open(self.configFile, 'r'), Loader=yaml.CLoader)

        self.frame_proc_q = mp.Queue(maxsize=-1)
        self.frame_hdf_q = None
        self.patch_q = mp.Queue(maxsize=-1)
        self.peak_hdf_q = None
        self.peak_zmq_q = None
        self.peak_pva_q = None

        params = self.params
        self.nFrameProcessors = params['frame']['nproc']
        self.nGpu = params['infer'].get('n_gpu', 2)
        self.logger.debug(f'Number of available GPUs: {self.nGpu}')

        # Create frame writer; receives data from frame processor
        self.frameHdfController = None
        if params['output']['frame2file']:
            self.frame_hdf_q = mp.Queue(maxsize=-1)
            self.frameHdfWriter = BraggNNHdfWriter('frame', fileName=params['output']['frame2file'], compression=True)
            self.frameHdfController = UserMpWorkerController(self.FRAME_HDF_WRITER_WORKER_ID, self.frameHdfWriter, self.frame_hdf_q)

        # Create frame processors; they send data to frame writer 
        self.frameProcControllerMap = {}
        for i in range(0,self.nFrameProcessors):
            workerId = f'{self.FRAME_PROCESSOR_WORKER_ID}.{i+1}'
            frameProcessor = BraggNNFrameProcessor(
                psz=params['model']['psz'],
                mbsz=params['infer']['mbsz'], 
                offset_recover=params['frame']['offset_recover'], 
                min_intensity=params['frame']['min_intensity'],
                max_radius=params['frame']['max_radius'],
                min_peak_sz=params['frame']['min_peak_sz'], 
                dark_h5=params['frame']['dark_h5'], 
                patch_q=self.patch_q, write_q=self.frame_hdf_q)
            self.frameProcControllerMap[i] = UserMpWorkerController(workerId, frameProcessor, self.frame_proc_q)

        # Create peak hdf writer; receives data from this processor
        self.peakHdfController = None
        if params['output']['peaks2file']:
            self.peak_hdf_q = mp.Queue(maxsize=-1)
            self.peakHdfWriter = BraggNNHdfWriter('peak', fileName=params['output']['peaks2file'], compression=False)
            self.peakHdfController = UserMpWorkerController(self.PEAK_HDF_WRITER_WORKER_ID, self.peakHdfWriter, self.peak_hdf_q)

        # Create peak zmq writer; receives data from this processor
        self.peakZmqController = None
        if params['output']['port4zmq']:
            self.peak_zmq_q = mp.Queue(maxsize=-1)
            self.peakZmqWriter = BraggNNZmqWriter(port=params['output']['port4zmq'])
            self.peakZmqController = UserMpWorkerController(self.PEAK_ZMQ_WRITER_WORKER_ID, self.peakZmqWriter, self.peak_zmq_q)

        # Stats
        self.nPatchBatchesProcessed = 0
        self.nPatchesPublished = 0
        self.inferTimeSum = 0
        self.publishTimeSum = 0

        self.isDone = False

    def _inferWorker(self):
        self.logger.debug('Starting infer worker')
        self.gpu = (self.processorId - 1) % self.nGpu
        self.logger.debug(f'Using gpu: {self.gpu}')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu)

        # Inference engine
        # params = self.params
        # if params['infer']['tensorrt']:
        #     from trtUtil import scriptpth2onnx
        #     from braggNNTrtInfer import BraggNNTrtInfer
        #     onnx_mdl = scriptpth2onnx(pth=params['model']['model_fname'], mbsz=params['infer']['mbsz'], psz=params['model']['psz'])
        #     inferEngine = BraggNNTrtInfer(onnx_mdl)
        # else:
        #     from braggNNTorchInfer import BraggNNTorchInfer
        #     inferEngine = BraggNNTorchInfer(script_pth=params['model']['model_fname'])

        while True:
            if self.isDone:
                break
            try:
                in_mb, ori_mb, frm_id = self.patch_q.get(block=True, timeout=self.Q_WAIT_TIME)
                # t0 = time.time()
                # pred = inferEngine.process(in_mb)
                # inferTime = time.time() - t0
                # self.inferTimeSum += inferTime
                ddict = {  # 'ploc' : np.concatenate([ori_mb, in_mb.shape[-1]], axis=1),
                    'patches' : in_mb,
                    'uniqueId' : frm_id
                }
                if self.peak_hdf_q:
                    self.peak_hdf_q.put(ddict)
                if self.peak_zmq_q:
                    self.peak_zmq_q.put(ddict)
                self.peak_pva_q.put(ddict)
                self.nPatchBatchesProcessed += 1
                self.logger.debug(f'Batch of {in_mb.shape[0]} patches; {self.patch_q.qsize()} batches pending infer.')
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.isDone = True
                break
            except EOFError:
                self.isDone = True
                break
            except Exception as ex:
                self.logger.error(f'Unexpected error caught: {ex} {type(ex)}')
                break

        try:
            self.logger.debug(f'Emptying patch queue, current size is {self.patch_q.qsize()}')
            while not self.patch_q.empty():
                self.patch_q.get(block=True, timeout=self.Q_WAIT_TIME)
            self.patch_q.close()
        except Exception as ex:
            self.logger.warn(f'Error emptying patch queue: {ex}')
        self.logger.debug('Infer worker is done')

    def _pvaPublishPeaks(self, ddict):
        seqId = 0
        frameId = ddict['uniqueId']
        nPatches = ddict['patches'].shape[0]
        self.logger.debug(f'Publishing {nPatches} patches for frame {frameId}')
        for i in range(nPatches):
            t0 = time.time()
            pdict = {}
            pdict['image'] = ddict['patches'][i]
            pdict['uniqueId'] = ddict['uniqueId']
            pdict['patchId'] = seqId # seqID

            seqId += 1

            a, ny, nx = pdict['image'].shape
            nda = pva.NtNdArray()
            meta = list(pdict.keys())[2:]
            attrs = [pva.NtAttribute(_key, pva.PvFloat(pdict[_key])) for _key in meta]
            nda['attribute'] = attrs
            nda['uniqueId'] = pdict['uniqueId']
            dims = [pva.PvDimension(nx, 0, nx, 1, False), 
                    pva.PvDimension(ny, 0, ny, 1, False)]
            nda['dimension'] = dims
            nda['descriptor'] = 'Patch'
            nda['value'] = {'intValue': np.array(pdict['image'].flatten(), dtype=np.int32)}
            self.updateOutputChannel(nda)
            publishTime = time.time()-t0
            self.publishTimeSum += publishTime
            self.nPatchesPublished += 1

    def _pvaWorker(self):
        self.logger.debug('Starting pva worker')
        while True:
            if self.isDone:
                break
            try:
                ddict = self.peak_pva_q.get(block=True, timeout=self.Q_WAIT_TIME)
                self._pvaPublishPeaks(ddict)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.isDone = True
                break
            except EOFError:
                self.isDone = True
                break
            except Exception as ex:
                self.logger.error(f'Unexpected error caught: {ex} {type(ex)}')
                break
        try:
            self.logger.debug(f'Emptying pva queue, current size is {self.peak_pva_q.qsize()}')
            while not self.peak_pva_q.empty():
                self.peak_pva_q.get(block=True, timeout=self.Q_WAIT_TIME)
            self.peak_pva_q.close()
        except Exception as ex:
            self.logger.warn(f'Error emptying pva queue: {ex}')
        self.logger.debug('Pva worker is done')

    def _getControllerStats(self):
        controllerStatsMap = {}
        for i in range(0,self.nFrameProcessors):
            procId = i+1
            cKey = f'{self.FRAME_PROCESSOR_WORKER_ID}{procId}'
            sd = self.frameProcControllerMap[i].getStats(statsKeyPrefix=f'{cKey}_')
            controllerStatsMap[cKey] = sd
        if self.frameHdfController:
            cKey = self.FRAME_HDF_WRITER_WORKER_ID
            sd = self.frameHdfController.getStats(statsKeyPrefix=f'{cKey}_')
            controllerStatsMap[cKey] = sd
        if self.peakHdfController:
            cKey = self.PEAK_HDF_WRITER_WORKER_ID 
            sd = self.peakHdfController.getStats(statsKeyPrefix=f'{cKey}_')
            controllerStatsMap[cKey] = sd
        if self.peakZmqController:
            cKey = self.PEAK_ZMQ_WRITER_WORKER_ID
            sd = self.peakZmqController.getStats(statsKeyPrefix=f'{cKey}_')
            controllerStatsMap[cKey] = sd
        return controllerStatsMap

    def _calculateStats(self, controllerStatsMap):
        nFramesProcessed = 0
        nPatchesGenerated = 0
        frameProcessingTimeSum = 0
        frameProcessingTime = 0
        frameProcessingRate = 0
        for i in range(0,self.nFrameProcessors):
            procId = i+1
            cKey = f'{self.FRAME_PROCESSOR_WORKER_ID}{procId}'
            sd = controllerStatsMap[cKey]
            nfp = sd.get(f'{cKey}_nFramesProcessed', 0)
            fpt = sd.get(f'{cKey}_processTime', 0)
            nFramesProcessed += nfp
            frameProcessingTimeSum += nfp*fpt
            npg = sd.get(f'{cKey}_nPatchesGenerated', 0)
            nPatchesGenerated += npg
        if nFramesProcessed > 0:
            frameProcessingTime = frameProcessingTimeSum/nFramesProcessed  
            frameProcessingRate = nFramesProcessed/frameProcessingTimeSum
        nFramesQueued = self.frame_proc_q.qsize()
        nPatchBatchesQueued = self.patch_q.qsize()

        # inferRate = 0
        # inferTime = 0
        # if self.nPatchBatchesProcessed > 0:
        #     inferRate = self.nPatchBatchesProcessed/self.inferTimeSum
        #     inferTime = self.inferTimeSum/self.nPatchBatchesProcessed

        publishRate = 0
        publishTime = 0
        if self.nPatchesPublished > 0:
            publishRate = self.nPatchesPublished/self.publishTimeSum
            publishTime = self.publishTimeSum/self.nPatchesPublished

        statsDict = {}
        statsDict['nFramesProcessed'] = nFramesProcessed
        statsDict['nFramesQueued'] = nFramesQueued
        statsDict['frameProcessingTime'] = frameProcessingTime
        statsDict['frameProcessingRate'] = frameProcessingRate
        statsDict['nPatchBatchesProcessed'] = self.nPatchBatchesProcessed
        statsDict['nPatchBatchesQueued'] = nPatchBatchesQueued
        # statsDict['inferTime'] = inferTime
        # statsDict['inferRate'] = inferRate
        statsDict['nPatchesGenerated'] = nPatchesGenerated
        statsDict['nPatchesPublished'] = self.nPatchesPublished
        statsDict['publishTime'] = publishTime
        statsDict['publishRate'] = publishRate

        for cKey,sd in controllerStatsMap.items():
            statsDict.update(sd)
        return statsDict

    def start(self):
        if self.frameHdfController:
            self.logger.debug('Starting frame HDF controller')
            self.frameHdfController.start()
        if self.peakHdfController:
            self.logger.debug('Starting peak HDF controller')
            self.peakHdfController.start()
        if self.peakZmqController:
            self.logger.debug('Starting peak ZMQ controller')
            self.peakZmqController.start()
        for i in range(0,self.nFrameProcessors):
            self.logger.debug(f'Starting frame processor {i+1}')
            self.frameProcControllerMap[i].start()
        self.inferThread = threading.Thread(target=self._inferWorker)
        self.inferThread.start()
        if self.outputChannel:
            self.peak_pva_q = mp.Queue(maxsize=-1)
            self.pvaThread = threading.Thread(target=self._pvaWorker)
            self.pvaThread.start()

    def stop(self):
        self.logger.debug('Signaling worker threads to stop')
        self.isDone = True
        controllerStatsMap = {}
        for i in range(0,self.nFrameProcessors):
            procId = i+1
            cKey = f'{self.FRAME_PROCESSOR_WORKER_ID}{procId}'
            self.logger.debug(f'Stopping frame processor {procId}')
            controllerStatsMap[cKey] = self.frameProcControllerMap[i].stop(statsKeyPrefix=f'{cKey}_')
        self.frame_proc_q.close()
        if self.frameHdfController:
            cKey = self.FRAME_HDF_WRITER_WORKER_ID
            self.logger.debug('Stopping frame HDF controller')
            controllerStatsMap[cKey] = self.frameHdfController.stop(statsKeyPrefix=f'{cKey}_')
            self.frame_hdf_q.close()
        if self.peakHdfController:
            cKey = self.PEAK_HDF_WRITER_WORKER_ID 
            self.logger.debug('Stopping peak HDF controller')
            controllerStatsMap[cKey] = self.peakHdfController.stop(statsKeyPrefix=f'{cKey}_')
            self.peak_hdf_q.close()
        if self.peakZmqController:
            cKey = self.PEAK_ZMQ_WRITER_WORKER_ID
            self.logger.debug('Stopping peak ZMQ controller')
            controllerStatsMap[cKey] = self.peakZmqController.stop(statsKeyPrefix=f'{cKey}_')
            self.peak_zmq_q.close()
        statsDict = self._calculateStats(controllerStatsMap)
        self.logger.debug('All controllers stopped, exiting')
        return statsDict

    def configure(self, kwargs):
        self.logger.debug(f'Configuration update: {kwargs}')

    def process(self, pvObject):
        if self.isDone:
            return
        frameId = pvObject['uniqueId']
        dims = pvObject['dimension']
        nx = dims[0]['size']
        ny = dims[1]['size']
        codec = pvObject['codec']
        compressedSize   = pvObject['compressedSize']
        uncompressedSize = pvObject['uncompressedSize']
        fieldKey = pvObject.getSelectedUnionFieldName()
        frameData = pvObject['value'][0][fieldKey]

        self.frame_proc_q.put((frameId, frameData, compressedSize, uncompressedSize, codec, ny, nx))
        return pvObject

    def resetStats(self):
        self.nPatchBatchesProcessed = 0
        self.nPatchesPublished = 0
        self.inferTimeSum = 0
        self.publishTimeSum = 0
        for i in range(0,self.nFrameProcessors):
            self.frameProcControllerMap[i].resetStats()
        if self.frameHdfController:
            self.frameHdfController.resetStats()
        if self.peakHdfController:
            self.peakHdfController.resetStats()
        if self.peakZmqController:
            self.peakZmqController.resetStats()

    # Retrieve statistics for user processor
    def getStats(self):
        controllerStatsMap = self._getControllerStats()
        return self._calculateStats(controllerStatsMap)

    # Define PVA types for different stats variables
    def getStatsPvaTypes(self):
        typeDict = {
            'nFramesProcessed' : pva.UINT,
            'nFramesQueued' : pva.UINT,
            'frameProcessingTime' : pva.DOUBLE,
            'frameProcessingRate' : pva.DOUBLE,
            'nPatchBatchesProcessed' : pva.UINT,
            'nPatchBatchesQueued' : pva.UINT,
            'inferTime' : pva.DOUBLE,
            'inferRate' : pva.DOUBLE,
            'nPatchesGenerated' : pva.UINT,
            'nPatchesPublished' : pva.UINT,
            'publishTime' : pva.DOUBLE,
            'publishRate' : pva.DOUBLE
        }
        for i in range(0,self.nFrameProcessors):
            procId = i+1
            typeDict[f'frameProcessor{procId}_nFramesProcessed'] = pva.UINT
            typeDict[f'frameProcessor{procId}_nPatchesGenerated'] = pva.UINT
            typeDict[f'frameProcessor{procId}_processTime'] = pva.DOUBLE
            typeDict[f'frameProcessor{procId}_decodeTime'] = pva.DOUBLE
            typeDict[f'frameProcessor{procId}_peakTime'] = pva.DOUBLE
        if self.frameHdfController:
            typeDict['frameHdfWriter_nObjectsWritten'] = pva.UINT
            typeDict['frameHdfWriter_writeTime'] = pva.DOUBLE
        if self.peakHdfController:
            typeDict['peakHdfWriter_nObjectsWritten'] = pva.UINT
            typeDict['peakHdfWriter_writeTime'] = pva.DOUBLE
        if self.peakZmqController:
            typeDict['peakZmqWriter_nObjectsPublished'] = pva.UINT
            typeDict['peakZmqWriter_nErrors'] = pva.UINT
            typeDict['peakZmqWriter_publishTime'] = pva.DOUBLE
        return typeDict

