from fastml.core import *
from fastml.data import *
from torch import optim

def SGD(model, lr=0.5):
    return optim.SGD(model.parameters(), lr=lr)