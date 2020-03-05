import torch.nn.functional as F
from torch import nn

from fastml.model import loss as loss
from fastml.model import metrics as metrics
from fastml.model import optimizers as opt
from fastml.data.datasets import Datasets


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, nh)
        self.l2 = nn.Linear(nh, n_out)

    def __call__(self, x): return self.l2(F.relu(self.l1(x)))


train_ds, valid_ds = Datasets.MNIST()
x_train, y_train = train_ds.x, train_ds.y

lr = 0.5
epochs = 5
batch_size = 64
loss_func = loss.cross_entropy()
model = Model(784, 50, 10)
optimizer = opt.SGD(model)

def fit():
    for epoch in range(epochs):
        for batch in range(x_train.shape[0] // batch_size):

            b_from = batch * batch_size
            b_to = b_from + batch_size
            x_batch = x_train[b_from:b_to]
            y_batch = y_train[b_from:b_to]

            loss = loss_func(model(x_batch), y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        accuracy = metrics.accuracy(model(x_train), y_train)
        print("epoch %s accuracy is %f3 loss is %f" % (epoch, accuracy, loss))


fit()
