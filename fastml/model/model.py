from fastml.core import *
from fastml.data.datasets import *
from fastml.model import loss as loss
from fastml.model import metrics as metrics
from fastml.model import optimizers
from fastml.model.callbacks import *
from torch import nn

class Learner():
    def __init__(self, model, optimizer, loss_func, data):
        self.model,self.optimizer,self.loss_func,self.data = model,optimizer,loss_func,data

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def optimizer(self): return self.learn.optimizer
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.train_with_batches(self.data.train_dl)
                with torch.no_grad():
                    if not self('begin_validate'): self.train_with_batches(self.data.valid_dl)
                self('after_epoch')
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def train_with_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl: self.train_with_one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def train_with_one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')

            if not self.in_train: return

            self.loss.backward()
            self('after_backward')
            self.optimizer.step()
            self('after_step')
            self.optimizer.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)