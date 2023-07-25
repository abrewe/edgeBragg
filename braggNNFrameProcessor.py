import numpy as np
import pvapy as pva
import cv2
import time
import h5py
from codecAD import CodecAD
from pvapy.hpc.userMpDataProcessor import UserMpDataProcessor

class BraggNNFrameProcessor(UserMpDataProcessor):

    def __init__(self, psz, mbsz, offset_recover, min_intensity, max_radius, min_peak_sz, dark_h5, patch_q, write_q):
        UserMpDataProcessor.__init__(self)
        self.psz = psz
        self.mbsz = mbsz
        self.offset_recover = offset_recover
        self.min_intensity = min_intensity
        self.max_radius = max_radius
        self.min_peak_sz = min_peak_sz
        self.dark_h5 = dark_h5
        self.dark_fr = self._getDarkFrame(self.dark_h5)
        self.patch_q = patch_q
        self.write_q = write_q
        self.codecAD = CodecAD()
        self.patch_list = []
        self.patch_ori_list = []
        self.resetStats()

    def _getDarkFrame(self, dark_h5):
        dark_fr = None
        if dark_h5 is not None:
            with h5py.File(dark_h5, 'r') as fp:
                dark_fr = fp['frames'][:]
            dark_fr = dark_fr[:].mean(axis=0)
        else:
            self.logger.debug('No dark h5 file supplied')
        return dark_fr

    # cv2 based geometric center connected component as center for crop
    def _framePeakPatchesCv2(self, frame, psz, angle, min_intensity=0, max_r=None, min_sz=1):
        fh, fw = frame.shape
        patches, peak_ori = [], []
        mask = (frame > min_intensity).astype(np.uint8)
        comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        big_peaks = 0
        small_peak = 0
        for comp in range(1, comps):
            # ignore too small peak
            area = stats[comp, cv2.CC_STAT_AREA]
            if stats[comp, cv2.CC_STAT_WIDTH] < min_sz or stats[comp, cv2.CC_STAT_HEIGHT] < min_sz:
                small_peak += 1
                continue
            
            # ignore component that is bigger than patch size
            if stats[comp, cv2.CC_STAT_WIDTH] > psz or stats[comp, cv2.CC_STAT_HEIGHT] > psz:
                big_peaks += 1
                continue
        
            # check if the component is within the max radius
            c, r = centroids[comp, 0], centroids[comp, 1]
            if max_r is not None and max_r**2 < ((c - fw/2)**2 + ( r - fh/2)**2):
                continue
                    
            col_s = stats[comp, cv2.CC_STAT_LEFT]
            col_e = col_s + stats[comp, cv2.CC_STAT_WIDTH]
        
            row_s = stats[comp, cv2.CC_STAT_TOP]
            row_e = row_s + stats[comp, cv2.CC_STAT_HEIGHT]

            _patch = frame[row_s:row_e, col_s:col_e]
        
            # mask out other labels in the patch
            _mask  = cc_labels[row_s:row_e, col_s:col_e] == comp
            _patch = _patch * _mask

            if _patch.size != psz * psz:
                h, w = _patch.shape
                _lp = (psz - w) // 2
                _rp = (psz - w) - _lp
                _tp = (psz - h) // 2
                _bp = (psz - h) - _tp
                _patch = np.pad(_patch, ((_tp, _bp), (_lp, _rp)), mode='constant', constant_values=0)
            else:
                _tp, _lp = 0, 0

            _min, _max = _patch.min(), _patch.max()
            if _min == _max: continue

            _pr_o = row_s - _tp
            _pc_o = col_s - _lp
            peak_ori.append((angle, _pr_o, _pc_o))
            patches.append(_patch)

        return patches, peak_ori, big_peaks


    def _processFrame(self, frm_id, data_codec, compressed, uncompressed, codec, rows, cols):
        startTick = time.time()
        self.logger.debug(f'Processing frame {frm_id}, codec: {codec}')
        if not codec['name']:
            data = data_codec 
        else:
            self.codecAD.decompress(data_codec, codec, compressed, uncompressed)
            data = self.codecAD.getData()
            decTime = startTick - time.time()
            self.decodeTimeSum += decTime
            self.logger.debug(f'frame {frm_id} has been decoded in {1000*decTime:.3f} ms using {codec["name"]}, compress ratio is {self.codecAD.getCompressRatio():.1f}')

        frame = data.reshape((rows, cols))

        # dark is not removed, thus remove here
        # min_intensity will deal with negative pixels
        if self.dark_fr is not None:
            frame = frame - self.dark_fr

        # dark was removed on EPICS server
        elif self.offset_recover != 0:
            frame[frame > 0] += self.offset_recover

        tick = time.time()
        patches, patch_ori, big_peaks = self._framePeakPatchesCv2(frame=frame, angle=frm_id, psz=self.psz, min_intensity=self.min_intensity, max_r=self.max_radius, min_sz=self.min_peak_sz)
        self.nPatchesGenerated += len(patches)
                                                               
        mbsz = self.mbsz
        # print("length of self.patchlist: " +str(len(self.patch_list)))
        # patch_list = self.patch_list
        # patch_ori_list = self.patch_ori_list
        patch_list = []
        patch_ori_list = []
        patch_list.extend(self.patch_list)
        patch_ori_list.extend(self.patch_ori_list)
        patch_q = self.patch_q
        write_q = self.write_q

        patch_list.extend(patches)
        patch_ori_list.extend(patch_ori)

        self.logger.debug(f'Patch list size is {len(patch_list)}, mbsz is {mbsz}')
        # while len(patch_list) >= mbsz:
        batch_task = (
                np.array(patches)[:,np.newaxis],
                np.array(patch_ori[:mbsz]).astype(np.float32),
                frm_id
        )



        patch_q.put(batch_task)
        patch_list = patch_list[mbsz:]
        patch_ori_list = patch_ori_list[mbsz:]

        self.patch_list = patch_list
        self.patch_ori_list = patch_ori_list

        peakTime = time.time() - tick
        self.peakTimeSum += peakTime
        self.logger.debug(f'{len(patches)} patches cropped from frame {frm_id}, {peakTime:.4f} seconds/frame; {big_peaks} peaks are too big')

        # back-up raw frames when required
        if write_q is not None:
            write_q.put({'angle':np.array([frm_id])[None], 'frame':frame[None]})
        processTime = time.time() - startTick
        self.processTimeSum += processTime
        self.nFramesProcessed += 1

    def process(self, mpqObject):
        frm_id, data_codec, compressed, uncompressed, codec, rows, cols = mpqObject
        self._processFrame(frm_id, data_codec, compressed, uncompressed, codec, rows, cols)

    def getStats(self):
        processTime = 0.0
        decodeTime = 0.0
        peakTime = 0.0
        if self.nFramesProcessed > 0:
            processTime = self.processTimeSum/self.nFramesProcessed
            decodeTime = self.decodeTimeSum/self.nFramesProcessed
            peakTime = self.peakTimeSum/self.nFramesProcessed
        statsDict = {
            'nFramesProcessed' : self.nFramesProcessed,
            'nPatchesGenerated' : self.nPatchesGenerated,
            'processTime' : processTime,
            'decodeTime' : decodeTime,  
            'peakTime' : peakTime
        }
        return statsDict

    def resetStats(self):
        self.nFramesProcessed = 0
        self.nPatchesGenerated = 0
        self.processTimeSum = 0.0
        self.decodeTimeSum = 0.0
        self.peakTimeSum = 0.0
