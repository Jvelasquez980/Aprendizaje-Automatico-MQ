import torch
import time

device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
a = torch.randn(12000, 12000, device=device)
b = torch.randn(12000, 12000, device=device)

torch.cuda.synchronize()
start = time.time()

c = torch.matmul(a, b)

torch.cuda.synchronize()
end = time.time()

print("Tiempo GPU:", end - start)