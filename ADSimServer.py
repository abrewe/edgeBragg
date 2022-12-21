#!/usr/bin/env python

import time, threading, queue, argparse
import numpy as np
# from PIL import Image
import os, os.path
import pvaccess as pva
import fabio

# modified version of ADSimServer.py by @vbanakha
__version__ = pva.__version__


class AdSimServer(threading.Thread):

    DELAY_CORRECTION = 0.0001
    PVA_TYPE_KEY_MAP = {
        np.dtype('uint8')   : 'ubyteValue',
        np.dtype('int8')    : 'byteValue',
        np.dtype('uint16')  : 'ushortValue',
        np.dtype('int16')   : 'shortValue',
        np.dtype('uint32')  : 'uintValue',
        np.dtype('int32')   : 'intValue',
        np.dtype('uint64')  : 'ulongValue',
        np.dtype('int64')   : 'longValue',
        np.dtype('float32') : 'floatValue',
        np.dtype('float64') : 'doubleValue'
    }

    def __init__(self, input_directory, frame_rate, nf, nx, ny, runtime, channel_name, start_delay, report_frequency):
        threading.Thread.__init__(self)
        self.arraySize = None
        self.delta_t = 0
        if frame_rate > 0:
            self.delta_t = 1.0/frame_rate
        self.runtime = runtime
        self.report_frequency = report_frequency
        self.in_directory = input_directory
        self.nx = nx
        self.ny = ny
        self.loaded_images = 0

        # input_files = []
        # if input_directory is not None:
        #     input_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
        # if input_file is not None:
        #     input_files.append(input_file)
        # contains collection of files to load into frames
        self.files = queue.Queue(maxsize=-1)
        self.nfiles = 0
        self.random_frames = False
        if self.in_directory is not None:
            self.get_files()
        else:
            print("No input directory: generating random frames")
            self.random_frames = True
        # frames from files for publishing
        # self.frames = queue.Queue(maxsize=-1)
        # self.n_input_frames = 0
        self.rows = 0
        self.cols = 0
        self.pva_type_key = None
        print('Number of input files: %s ' % (self.nfiles))
            # self.frames = np.random.randint(0, 256, size=(nf, nx, ny), dtype=np.int16)
        # self.cont = True
        # i = 0
        # for f in input_files:
        #     i += 1
        #     try:
        #         im = Image.open(f)
        #         new_frames = np.array(im)
        #         new_frames = np.expand_dims(new_frames, axis=0)
        #         if self.frames.empty():
        #             self.frames = self.frames.put(new_frames)
        #         else:
        #             self.frames = self.frames.put(new_frames)
        #         print('Loaded input file %s' % (f))
        #     except Exception as ex:
        #         print('Cannot load input file %s, skipping it: %s' % (f, ex))
        #     self.pva_type_key = self.PVA_TYPE_KEY_MAP.get(self.frames.dtype)
        # if self.frames is None:
        #         print('No frames loaded')

        # self.frames = np.random.randint(0, 256, size=(nf, nx, ny), dtype=np.int16)
        # self.n_input_frames, self.rows, self.cols = self.frames.shape
        # self.n_input_frames, self.cols, self.rows = self.frames.shape
        # self.pva_type_key = self.PVA_TYPE_KEY_MAP.get(self.frames.dtype)

        self.channel_name = channel_name
        self.frame_rate = frame_rate
        self.server = pva.PvaServer()
        self.server.addRecord(self.channel_name, pva.NtNdArray())
        # self.frame_map = {}
        self.current_frame_id = 0
        self.n_published_frames = 0
        self.start_time = 0
        self.last_published_time = 0
        self.start_delay = start_delay
        self.is_done = False

    def get_files(self):
        input_files = []
        if self.in_directory is not None:
            input_files = [os.path.join(self.in_directory, f) for f in os.listdir(self.in_directory) if os.path.isfile(os.path.join(self.in_directory, f))]
        if len(input_files) != 0:
            self.nfiles = len(input_files)
        for f in input_files:
            self.files.put(f)

    def get_timestamp(self):
        s = time.time()
        ns = int((s-int(s))*1000000000)
        s = int(s)
        return pva.PvTimeStamp(s,ns)

    def frame_producer(self, frame, id, extraFieldsPvObject=None):
        # for frame_id in range(0, self.n_input_frames):
        if self.is_done:
            return

        if extraFieldsPvObject is None:
             nda = pva.NtNdArray()
        else:
            nda = pva.NtNdArray(extraFieldsPvObject.getStructureDict())


        nda['uniqueId'] = id
        nda['codec'] = pva.PvCodec('pvapyc', pva.PvInt(5))
        # nda['codec'] = pva.PvCodec('', pva.PvInt(5))
        dims = [pva.PvDimension(self.rows, 0, self.rows, 1, False), \
                pva.PvDimension(self.cols, 0, self.cols, 1, False)]
        nda['dimension'] = dims
        nda['compressedSize'] = self.rows*self.cols
        nda['uncompressedSize'] = self.rows*self.cols
        ts = self.get_timestamp()
        nda['timeStamp'] = ts
        nda['dataTimeStamp'] = ts
        nda['descriptor'] = 'PvaPy Simulated Image'
        # nda['value'] = {self.pva_type_key : self.frames[frame_id].flatten()}
        nda['value'] = {self.pva_type_key: frame.flatten()}
        attrs = [pva.NtAttribute('ColorMode', pva.PvInt(0))]

        nda['attribute'] = attrs
        if extraFieldsPvObject is not None:
            nda.set(extraFieldsPvObject)
        # self.frame_map[frame_id] = nda
        return nda

    def prepare_frame(self):
        # Get cached frame
        # cached_frame_id = self.current_frame_id % self.n_input_frames
        # frame = self.frame_map[cached_frame_id]
        if self.files.empty() and self.loaded_images == 0 and not self.random_frames:
            print("hello")
            print("No image files loaded")
            self.stop()
        elif self.files.empty() and self.in_directory is not None:
            self.loaded_images = 0
            self.get_files()
        try:
            if self.random_frames:
                frame = np.random.randint(0, 256, size=(self.nx, self.ny), dtype=np.int16)
            else:
                file = self.files.get()
                frame = fabio.open(file).data
                print('Loaded input file %s' % (file))
                self.loaded_images += 1
            self.cols, self.rows = frame.shape
            # frame = np.array(im)
        except Exception as ex:
            print('Cannot load input file %s, skipping it: %s' % (file, ex))
            return None

        if frame is not None:
            # Correct image id and timestamps
            self.current_frame_id += 1
            self.pva_type_key = self.PVA_TYPE_KEY_MAP.get(frame.dtype)
            if self.pva_type_key is None:
                return
            frame = self.frame_producer(frame, self.current_frame_id)
            frame['uniqueId'] = self.current_frame_id
            ts = self.get_timestamp()
            frame['timeStamp'] = ts
            frame['dataTimeStamp'] = ts
            return frame

    def frame_publisher(self):
        while True:
            if self.is_done:
                return
            frame = None
            while frame is None:
                frame = self.prepare_frame()
            # frame = self.prepare_frame()
            # if frame is None:
            #     return
            self.server.update(self.channel_name, frame)
            self.last_published_time = time.time()
            self.n_published_frames += 1

            runtime = 0
            if self.n_published_frames > 1:
                runtime = self.last_published_time - self.start_time
                delta_t = runtime/(self.n_published_frames - 1)
                frame_rate = 1.0/delta_t
                if self.report_frequency > 0 and (self.n_published_frames % self.report_frequency) == 0:
                    print("Published frame id %6d @ %.3f (frame rate: %.4f fps)" % (self.current_frame_id, self.last_published_time, frame_rate))
            else:
                self.start_time = self.last_published_time
                if self.report_frequency > 0 and (self.n_published_frames % self.report_frequency) == 0:
                    print("Published frame id %6d @ %.3f" % (self.current_frame_id, self.last_published_time))

            if runtime > self.runtime:
                print("Server will exit after reaching runtime of %s seconds" % (self.runtime))
                return

            if self.delta_t > 0:
                next_publish_time = self.start_time + self.n_published_frames*self.delta_t
                delay = next_publish_time - time.time() - self.DELAY_CORRECTION
                if delay > 0:
                    threading.Timer(delay, self.frame_publisher).start()
                    return

    def start(self):
        # threading.Thread(target=self.frame_producer, daemon=True).start()
        self.server.start()
        threading.Timer(self.start_delay, self.frame_publisher).start()

    def stop(self):
        self.is_done = True
        self.server.stop()
        runtime = self.last_published_time - self.start_time
        delta_t = runtime/(self.n_published_frames - 1)
        frame_rate = 1.0/delta_t
        print('\nServer runtime: %.4f seconds' % (runtime))
        print('Published frames: %6d @ %.4f fps' % (self.n_published_frames, frame_rate))

def main():
    parser = argparse.ArgumentParser(description='PvaPy Area Detector Simulator')
    parser.add_argument('--input-directory', '-id', type=str, dest='input_directory', default=None, help='Directory containing input files to be streamed; if input directory or input file are not provided, random images will be generated')
    # parser.add_argument('--input-file', '-if', type=str, dest='input_file', default=None, help='Input file to be streamed; if input directory or input file are not provided, random images will be generated')
    parser.add_argument('--frame-rate', '-fps', type=float, dest='frame_rate', default=20, help='Frames per second (default: 20 fps)')
    parser.add_argument('--n-x-pixels', '-nx', type=int, dest='n_x_pixels', default=256, help='Number of pixels in x dimension (default: 256 pixels; does not apply if input_file file is given)')
    parser.add_argument('--n-y-pixels', '-ny', type=int, dest='n_y_pixels', default=256, help='Number of pixels in x dimension (default: 256 pixels; does not apply if hdf5 file is given)')
    parser.add_argument('--n-frames', '-nf', type=int, dest='n_frames', default=1000, help='Number of different frames to generate and cache; those images will be published over and over again as long as the server is running')
    parser.add_argument('--runtime', '-rt', type=float, dest='runtime', default=300, help='Server runtime in seconds (default: 300 seconds)')
    parser.add_argument('--channel-name', '-cn', type=str, dest='channel_name', default='simulation:pva:test', help='Server PVA channel name (default: simulation:pva:test)')
    parser.add_argument('--start-delay', '-sd', type=float, dest='start_delay',  default=3.0, help='Server start delay in seconds (default: 3 seconds)')
    parser.add_argument('--report-frequency', '-rf', type=int, dest='report_frequency', default=1, help='Reporting frequency for publishing frames; if set to <=0 no frames will be reported as published (default: 1)')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 0:
        print('Unrecognized argument(s): %s' % ' '.join(unparsed))
        exit(1)

    # server = AdSimServer(input_directory=args.input_directory, input_file=args.input_file, frame_rate=args.frame_rate, nf=args.n_frames, nx=args.n_x_pixels, ny=args.n_y_pixels, runtime=args.runtime, channel_name=args.channel_name, start_delay=args.start_delay, report_frequency=args.report_frequency)
    server = AdSimServer(input_directory=args.input_directory, frame_rate=args.frame_rate, nf=args.n_frames, nx=args.n_x_pixels, ny=args.n_y_pixels, runtime=args.runtime, channel_name=args.channel_name, start_delay=args.start_delay, report_frequency=args.report_frequency)

    server.start()
    try:
        runtime = args.runtime + 2*args.start_delay
        time.sleep(runtime)
    except KeyboardInterrupt as ex:
        pass
    server.stop()

if __name__ == '__main__':
    main()