import torch
import numpy as np
from pvapy.utility.loggingManager import LoggingManager

class BraggNNTorchInfer:

    def __init__(self, script_pth):
        self.logger = LoggingManager.getLogger(self.__class__.__name__)
        self.torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.torch_dev = torch.device('cuda')
            self.braggNN = torch.jit.load(script_pth, map_location='cuda:0')
        else:
            self.torch_dev = torch.device('cpu')
            self.braggNN = torch.jit.load(script_pth, map_location='cpu')
        self.psz = self.braggNN.input_psz.item()
        # self.braggNN = torch.jit.freeze(self:.braggNN.eval())
        # self.braggNN = torch.jit.optimize_for_inference(self.braggNN) 

        self.logger.debug('PyTorch Inference engine initialization completed')

    def process(self, in_mb):
        input_tensor = torch.from_numpy(in_mb.astype(np.float32))
        with torch.no_grad():
            pred = self.braggNN.forward(input_tensor.to(self.torch_dev)).cpu().numpy()
        return pred

    def stop(self):
        pass
