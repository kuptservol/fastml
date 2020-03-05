from fastml.core import *
from fastml.data import *
from torch import optim


def SGD(model, lr=0.5):
    return optim.SGD(model.parameters(), lr=lr)


def Adam(model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
