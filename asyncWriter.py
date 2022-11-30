import h5py, logging, zmq, queue, threading, time
from multiprocessing import Process, Queue
import numpy as np
import pvaccess as pva
from collections import OrderedDict


class asyncHDFWriter(threading.Thread):
    def __init__(self, fname, compression=False):
        threading.Thread.__init__(self)
        self.daemon = True
        self.fname = fname
        self.h5fd = None
        self.task_q = Queue(maxsize=-1)
        self.compression = compression
    '''
    Args:
        ddict: dict of datasets to be written to h5, data will be concatenated on
               the first dimension
    '''
    def append2write(self, ddict):
        self.task_q.put(ddict)

    def run(self,):
        logging.info(f"Async writer to {self.fname} started ...")
        while True:
            ddict = self.task_q.get()
            if self.h5fd is None:
                self.h5fd = h5py.File(self.fname, 'w')
                for key, data in ddict.items():
                    dshape = list(data.shape)
                    dshape[0] = None
                    if self.compression:
                        self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape, compression="gzip")
                    else:
                        self.h5fd.create_dataset(key, data=data, chunks=True, maxshape=dshape)
                    logging.info(f"{data.shape} samples added to '{key}' of {self.fname}")
            else:
                for key, data in ddict.items():
                    self.h5fd[key].resize((self.h5fd[key].shape[0] + data.shape[0]), axis=0)
                    self.h5fd[key][-data.shape[0]:] = data
                    logging.info(f"{data.shape} samples added to '{key}' of {self.fname}, now has {self.h5fd[key].shape}")
            self.h5fd.flush()

'''
as zmq socket is not pickable, Thread, instead of Process should be used
or move the socket creation to the sending function before loop
'''


class asyncZMQWriter(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.daemon = True
        self.port = port
        self.task_q = queue.Queue(maxsize=-1)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(f"tcp://*:{self.port}")

    def append2write(self, ddict):
        self.task_q.put(ddict)

    def run(self,):
        logging.info(f"Async writer to ZMQ:{self.port} started ...")
        while True:
            ddict = self.task_q.get()
            ret = self.publisher.send_pyobj(ddict)
            logging.info(f"datasets {ddict.keys()} have been published via ZMQ {ret}")


class asyncPVAPub(threading.Thread):
    def __init__(self, channel, freq):
        threading.Thread.__init__(self)
        self.daemon = True
        self.freq = freq # maximum messages per second
        self.channel = channel

        self.server = pva.PvaServer()
        self.server.start()

        self.task_q = queue.Queue(maxsize=-1)

        self.first_msg = True

    def append2write(self, ddict):
        seq_id = 0
        for i in range(ddict["ploc"].shape[0]):
            pdict = OrderedDict()
            pdict['image'] = ddict['patches'][i]
            pdict['uniqueId'] = ddict['uniqueId']
            pdict['loc_fy'] = ddict["ploc"][i, 1] + ddict["ploc"][i, 3]
            pdict['loc_fx'] = ddict["ploc"][i, 2] + ddict["ploc"][i, 4]
            pdict['loc_py'] = ddict["ploc"][i, 3]
            pdict['loc_px'] = ddict["ploc"][i, 4]
            pdict['patchId'] = seq_id
            seq_id += 1
            self.task_q.put(pdict)
        logging.info(f"message {ddict['uniqueId']} publishing")

    def run(self):
        self.server.start()
        threading.Timer(0, self.msg_pub).start()
        logging.info(f"Async PVA writer to {self.channel} started ...")

    def msg_pub(self):
        ddict = self.task_q.get()
        proc_tick = time.time()
        a, r, c = ddict['image'].shape
        nda = pva.NtNdArray()

        meta = list(ddict.keys())[2:]
        attrs = [pva.NtAttribute(_key, pva.PvFloat(ddict[_key])) for _key in meta]

        nda['attribute'] = attrs
        nda['uniqueId'] = ddict['uniqueId']
        # nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(14))
        dims = [pva.PvDimension(r, 0, r, 1, False), \
                pva.PvDimension(c, 0, c, 1, False)]
        nda['dimension'] = dims
        nda['descriptor'] = 'Bragg Peak'
        nda['value'] = {'intValue': np.array(ddict['image'].flatten(), dtype=np.int32)}

        if self.first_msg:
            self.first_msg = False
            self.server.addRecord(self.channel, nda)
            time.sleep(1)  # give some time to propagate
        else:
            self.server.update(self.channel, nda)

        delay = max(0, 1.0/self.freq - (time.time() - proc_tick))
        threading.Timer(delay, self.msg_pub).start()


