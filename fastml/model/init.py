from fastml.core import *
from fastml.data import *

def kaiming_normal(num_in, num_out):
    w = torch.randn(num_in,num_out) * math.sqrt(2/num_in)
    b = torch.zeros(num_out)
    return w, b