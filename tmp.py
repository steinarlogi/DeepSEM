import torch
import torch.nn.functional as F

a = torch.randn(size=(32, 100, 256))
p = F.one_hot(torch.arange(0, 32), 100)

p = p[:, :, None].expand(-1, -1, 256) != 0

print (p.shape)
print (a.shape)

print (a * p + a * ~p)

print (torch.allclose(a, a * p + a * ~p))