from fastml.core import *
from fastml.data import *

def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()

def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()

def log_softmax(x): return x - x.logsumexp(-1,keepdim=True)

def cross_entropy(): return F.cross_entropy