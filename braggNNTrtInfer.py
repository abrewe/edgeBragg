import numpy as np
from pvapy.utility.loggingManager import LoggingManager

class BraggNNTrtInfer:
    def __init__(self, onnx_mdl):
        self.logger = LoggingManager.getLogger(self.__class__.__name__)
        self.onnx_mdl = onnx_mdl

        import tensorrt as trt
        from trtUtil import engine_build_from_onnx, mem_allocation
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        self.context = pycuda.autoinit.context
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()
        self.logger.debug('TensorRT Inference Engine initialization completed')

    def process(self, in_mb):
        from trtUtil import inference
        np.copyto(self.trt_hin, in_mb.astype(np.float32).ravel())
        pred = inference(self.trt_context, self.trt_hin, self.trt_hout, \
                         self.trt_din, self.trt_dout, self.trt_stream).reshape(-1, 2)
        return pred

    def stop(self):
        try:
            self.context.pop()
        except Exception as ex:
            pass

