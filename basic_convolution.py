import autograd.numpy as np
from autograd import grad
from params import Params
import nn_base 
import nn_types
import nn_helpers
import sys
import os
from tqdm import tqdm
import mnist


# Load mnist.
training_data, training_labels = mnist.load_mnist('training', path=os.getcwd())
validation_data, validation_labels = mnist.load_mnist('testing', path=os.getcwd())
DATA_SIZE = 784
NUM_CLASSES = 10
DROPOUT_RATE = 0.1


# Network construction.
network = nn_base.NNetwork()
network.set_input_shape((-1, DATA_SIZE))
# network.add_node(nn_types.ReshapeNode((-1, 1, 28, 28)))


# network.add_node(nn_types.DropoutNode(DROPOUT_RATE))
# network.add_node(nn_types.ConvolutionNode(8, 3))
# # TODO Normalize
# network.add_node(nn_types.ReluNode())
# network.add_node(nn_types.MaxPoolNode())

# network.add_node(nn_types.DropoutNode(DROPOUT_RATE))
# network.add_node(nn_types.ConvolutionNode(16, 4))
# # TODO Normalize
# network.add_node(nn_types.ReluNode())
# network.add_node(nn_types.MaxPoolNode())

# #16x5x5
# network.add_node(nn_types.DropoutNode(DROPOUT_RATE))
# network.add_node(nn_types.ConvolutionNode(16, 5))
# # TODO Normalize

#network.add_node(nn_types.ReshapeNode((-1, 16)))
# network.add_node(nn_types.ReshapeNode((-1, 16*5*5)))
network.add_node(nn_types.PerceptronLayer(100))
network.add_node(nn_types.ReluNode())
network.add_node(nn_types.PerceptronLayer(10))


network.add_node(nn_types.LogsoftmaxNode())


# Parameters construction
weights = nn_helpers.create_random_noise_array(
        network.get_params_shape())

# Create functions
cost_f, validating_f, _ = nn_helpers.create_functions(network)

print "Error rate: ", validating_f(weights, validation_data, validation_labels)
def epoch():
    global weights
    nn_helpers.BATCH_SIZE = 50
    batches = nn_helpers.create_batches(training_data, training_labels)
    for batch in tqdm(batches):
        weights = nn_helpers.train_on_batch(weights, *batch, cost_f=cost_f)
    print "Error rate: ", validating_f(weights, validation_data, validation_labels)

for i in range(50):
    epoch()





