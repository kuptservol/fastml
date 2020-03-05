import torch
import torch.nn.functional as F
from torch import nn

from fastml.examples.examples import get_mnist_data_bunch
from fastml.model import loss as loss
from fastml.model import metrics as metrics
from fastml.model import optimizers as opt


class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in, nh)
        self.l2 = nn.Linear(nh, n_out)

    def __call__(self, x): return self.l2(F.relu(self.l1(x)))


data_bunch = get_mnist_data_bunch()

lr = 0.5
epochs = 5
loss_func = loss.cross_entropy()
model = Model(784, 50, 10)
optimizer = opt.Adam(model)

def fit():
    for epoch in range(epochs):
        for (x_batch, y_batch) in data_bunch.train_dl:

            loss = loss_func(model(x_batch), y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            total_accuracy, total_loss = 0., 0.
            for x_valid_batch, y_valid_batch in data_bunch.valid_dl:
                valid_preds = model(x_valid_batch)
                total_loss += loss_func(valid_preds, y_valid_batch)
                total_accuracy += metrics.accuracy(valid_preds, y_valid_batch)

        n = len(data_bunch.valid_dl)
        print("epoch %s validation accuracy: %f3 loss: %f3" % (epoch, total_accuracy / n, total_loss.item() / n))


fit()
