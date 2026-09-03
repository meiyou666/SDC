import torch
import torch.nn as nn
import numpy as np
import math
from fault_inject import bitflip
from Detection0D import col_detect0D

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

# 滞后检测降低检测开销(产生nondeterministic不是增大检测开销吗)
# AS protection section
chk1 = Q.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (Q * weights).sum(axis=0)
Q_aug = np.vstack([Q, chk1, chk2]).astype(np.float32)

chk1 = K.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (K * weights).sum(axis=0)
K_aug = np.vstack([K, chk1, chk2]).astype(np.float32)

Q_faulty = bitflip(Q)
chk1 = Q_faulty.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (Q_faulty * weights).sum(axis=0)
Q_faulty_aug = np.vstack([Q_faulty, chk1, chk2]).astype(np.float32)

AS_aug = Q_aug @ K_aug.T
AS_faulty_aug = Q_faulty_aug @ K_aug.T # nondeterministic
col_detect0D(AS_aug, AS_faulty_aug)

# CL protection section
chk1 = AP.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (AP * weights).sum(axis=0)
AP_aug = np.vstack([AP, chk1, chk2]).astype(np.float32)

chk1 = V.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (V * weights).sum(axis=1, keepdims=True)
V_aug = np.hstack([V, chk1, chk2]).astype(np.float32)

V_faulty = bitflip(V)
chk1 = V_faulty.sum(axis=1, keepdims=True)
weights = np.arange(1, 9, dtype=np.float32).reshape(1, 8)
chk2 = (V_faulty * weights).sum(axis=1, keepdims=True)
V_faulty_aug = np.hstack([V_faulty, chk1, chk2]).astype(np.float32)

CL_aug = AP_aug @ V_aug
CL_faulty_aug = AP_aug @ V_faulty_aug 
col_detect0D(CL_aug, CL_faulty_aug)

# O protection section 
O_aug = CL_aug @ WQ
O_faulty = bitflip(O)
chk1 = O_faulty.sum(axis=0)
weights = np.arange(1, 9, dtype=np.float32).reshape(8, 1)
chk2 = (O_faulty * weights).sum(axis=0)
O_faulty_aug = np.vstack([O_faulty, chk1, chk2]).astype(np.float32)
col_detect0D(O_aug, O_faulty_aug)