from fastml.core import *
from fastml.data import *
from torch import optim

class SGDOptimizer():
    def __init__(self, params, lr=0.5): self.params,self.lr=list(params),lr

    def step(self):
        with torch.no_grad():
            for p in self.params: p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()

def SGD(model):
    return SGDOptimizer(model.parameters())