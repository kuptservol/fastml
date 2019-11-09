from fastml.core import *
from fastml.data import *
from fastml.examples import examples
from fastml.model.model import *
from fastml.model.image.cnn import *

mnist_data = examples.get_mnist_data_bunch()

runner, learner = examples.get_mnist_cnn(mnist_data)
runner.fit(1, learner)