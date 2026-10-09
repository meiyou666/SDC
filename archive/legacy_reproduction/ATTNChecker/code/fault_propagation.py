import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from fault_inject import bitflip

np.random.seed(42)

X = np.random.randn(8, 8).astype(np.float32)
WQ = np.random.randn(8, 8).astype(np.float32)
WK = np.random.randn(8, 8).astype(np.float32)
WV = np.random.randn(8, 8).astype(np.float32)

Q = X @ WQ
K = X @ WK
V = X @ WV

AS = Q @ K.T
d_k = K.shape[-1]
scores = AS / math.sqrt(d_k)
softmax = nn.Softmax(dim=-1)
AP = softmax(torch.tensor(scores))

CL = AP.detach().numpy() @ V
O = CL @ WQ

# Fault Injection
# Q = bitflip(Q)
# K = bitflip(K)
# V = bitflip(V)
# AS = bitflip(AS)
# CL = bitflip(CL)

# Fault Propagation_1
Q_faulty = bitflip(Q)
AS_faulty = Q_faulty @ K
error_1 = np.abs(Q_faulty - Q)
error_2 = np.abs(AS_faulty - AS)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(error_1, cmap='viridis')
axes[0].set_title("0D")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(error_2, cmap='viridis')
axes[1].set_title("1R")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Fault Propagation_2
V_faulty = bitflip(V)
CL_faulty = AP.detach().numpy() @ V_faulty
O_faulty = CL_faulty @ WQ
error_3 = np.abs(V_faulty - V)
error_4 = np.abs(CL_faulty - CL)
error_5 = np.abs(O_faulty - O)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

im1 = axes[0].imshow(error_3, cmap='viridis')
axes[0].set_title("0D")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(error_4, cmap='viridis')
axes[1].set_title("1C")
fig.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(error_5, cmap='viridis')
axes[2].set_title("2D")
fig.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()