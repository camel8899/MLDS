import torch
from torch.autograd import Variable

a = Variable(torch.rand(2,3))
print(a)
m = torch.nn.LogSoftmax()
print(m(a))
print(torch.sum(torch.exp(m(a)),1))
