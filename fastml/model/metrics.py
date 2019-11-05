from fastml.core import *
from fastml.data import *

def accuracy(out, yb): return 100*(torch.argmax(out, dim=1)==yb).float().mean().item()