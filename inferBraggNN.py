import logging, time, threading, torch
import numpy as np

class inferBraggNNtrt(threading.Thread):
    def __init__(self, mbsz, onnx_mdl, tq_patch, peak_writer, zmq_writer=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tq_patch = tq_patch
        self.mbsz = mbsz
        self.onnx_mdl = onnx_mdl
        self.writer = peak_writer
        self.zmq_writer = zmq_writer

    def run(self, ):
        from trtUtil import engine_build_from_onnx, mem_allocation, inference
        import pycuda.autoinit # must be in the same thread as the actual cuda execution
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, \
            self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()
        logging.info("TensorRT Inference engine initialization completed!")

        while True:
            in_mb, ori_mb, frm_id = self.tq_patch.get()
            batch_tick = time.time()
            np.copyto(self.trt_hin, in_mb.astype(np.float32).ravel())
            comp_tick  = time.time()
            pred = inference(self.trt_context, self.trt_hin, self.trt_hout, \
                             self.trt_din, self.trt_dout, self.trt_stream).reshape(-1, 2)
            t_comp  = 1000 * (time.time() - comp_tick)
            t_batch = 1000 * (time.time() - batch_tick)
            logging.info("A batch of %d patches was infered in %.3f ms (computing: %.3f ms), %d batches pending infer." % (\
                         self.mbsz, t_batch, t_comp, self.tq_patch.qsize()))

            ddict = {"ploc":np.concatenate([ori_mb, pred*in_mb.shape[-1]], axis=1), \
                     "patches":in_mb, "uniqueId": frm_id}
            self.writer.append2write(ddict)
            if self.zmq_writer is not None:
                self.zmq_writer.append2write(ddict)

class inferBraggNNTorch(threading.Thread):
    def __init__(self, script_pth, tq_patch, peak_writer, zmq_writer=None):
        threading.Thread.__init__(self)
        self.daemon = True
        self.tq_patch = tq_patch
        self.torch_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.BraggNN = torch.jit.load(script_pth, map_location='cuda:0')
        else:
            self.BraggNN = torch.jit.load(script_pth, map_location='cpu')
        self.psz = self.BraggNN.input_psz.item()
        # self.BraggNN = torch.jit.freeze(self.BraggNN.eval())
        # self.BraggNN = torch.jit.optimize_for_inference(self.BraggNN) # still in PyTroch prototype

        self.writer = peak_writer
        self.zmq_writer = zmq_writer
        logging.info("PyTorch Inference engine initialization completed!")

    def run(self, ):
        while True:
            in_mb, ori_mb, frm_id = self.tq_patch.get()
            batch_tick = time.time()
            input_tensor = torch.from_numpy(in_mb.astype(np.float32))
            comp_tick = time.time()
            with torch.no_grad():
                pred = self.BraggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
            t_comp  = 1000 * (time.time() - comp_tick)
            t_batch = 1000 * (time.time() - batch_tick)
            logging.info("A batch of %d patches infered in %.3f ms (computing: %.3f ms), %d batches pending infer." % (\
                         pred.shape[0], t_batch, t_comp, self.tq_patch.qsize()))

            ddict = {"ploc":np.concatenate([ori_mb, pred*in_mb.shape[-1]], axis=1), \
                     "patches":in_mb, "uniqueId":frm_id}

            self.writer.append2write(ddict)
            if self.zmq_writer is not None:
                self.zmq_writer.append2write(ddict)