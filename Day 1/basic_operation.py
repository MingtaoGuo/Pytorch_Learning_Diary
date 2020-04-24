import torch
import numpy as np

a = torch.rand(4, 5, requires_grad=True)
b = torch.ones(2, 3)
print(b)
c = a + 1
torch.mean(c)

a = torch.randn(4, 4)
b = a * 3
c = b.mean()
print(c.requires_grad)
c.requires_grad_(True)
print(c.requires_grad)

a = torch.ones(5, 5, requires_grad=True)
b = torch.randn(5, 5, requires_grad=True)
c = a + b
d = a * b
cc = c.mean()
dd = d.mean()

pass