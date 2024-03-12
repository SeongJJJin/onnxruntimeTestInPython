import torch
import onnxruntime

print("PyTorch version: {}".format(torch.__version__))
print("CUDA version: {}".format(torch.version.cuda))
print("cuDNN version: {}".format(torch.backends.cudnn.version()))
print("Onnxruntime version: {}".format(onnxruntime.__version__))
print(torch.cuda.get_device_name(0))