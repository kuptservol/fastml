from fastml.model.model import *
from fastml.model.image import *
from fastml.model.callbacks import *

def get_mnist_data_bunch(batch_size=64):
    train_ds,valid_ds = Datasets.MNIST()
    return data_bunch(train_ds, valid_ds, batch_size, train_ds.y.max().item()+1)

def get_simple_mnist_lin_learner(data, lr=0.3, nh=50):
    out_classes = data.train_ds.y.max().item()+1
    in_ = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(in_,nh), nn.ReLU(), nn.Linear(nh,out_classes))
    return Learner(model, optimizers.SGD(model, lr=lr), loss.cross_entropy(), data)

def get_mnist_cnn(mnist_data, lr=0.3, cnn_layers=[8,16,32,32], loss=loss.cross_entropy(), set_opt_F=lambda model, lr: optimizers.SGD(model, lr), cb_funcs=[]):

    mnist_view = view_tfm(1,28,28)
    [cb_funcs.append(cb) for cb in [Recorder, partial(AvgStatsCallback, metrics.accuracy), partial(BatchTransformXCallback, mnist_view)]]
    model = get_cnn_model(mnist_data, cnn_layers)
    opt = set_opt_F(model, lr)
    learner =  Learner(model, opt, loss, mnist_data)
    runner = Runner(cb_funcs=cb_funcs)

    return runner, learner