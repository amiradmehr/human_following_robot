import torch
import detectron2

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Detectron2 version:", detectron2.__version__)
