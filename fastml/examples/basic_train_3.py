import torch
import torch.nn.functional as F
from torch import nn

from fastml.examples.examples import get_mnist_data_bunch, Learner
from fastml.model import loss as loss
from fastml.model import metrics as metrics
from fastml.model import optimizers as opt
from fastml.model.callbacks import AvgStatsCallback
from fastml.model.model import Runner

data = get_mnist_data_bunch(batch_size=64)

model = nn.Sequential(nn.Linear(28 * 28, 100),
                      nn.ReLU(),
                      nn.Linear(100, 100),
                      nn.ReLU(),
                      nn.Linear(100, 10))

learner = Learner(model, opt.Adam(model), loss.cross_entropy(), data)

stats = AvgStatsCallback([metrics.accuracy])
run = Runner(cbs=stats)

run.fit(5, learner)
