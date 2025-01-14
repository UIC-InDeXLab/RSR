import torch

t1 = torch.load("weight-v1.0.pt")
t2 = torch.load("weight-v1.1.pt")

# print(torch.equal(t1, t2))
print(t2)