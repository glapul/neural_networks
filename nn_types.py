from params import Params
from nn_base import NNNode, NNetwork, ZeroArgNode
from operator import mul
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.scipy.signal import convolve

def compare_shapes(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != -1 and  b[i] != -1 and a[i] != b[i]:
            return False
    return True

def shape_assertion(func):
    def wrapper(self, data, params):
        if not compare_shapes(self.input_shape, data.shape):
            print type(self), self.input_shape, data.shape
            assert False
        res = func(self, data, params)
        if not compare_shapes(res.shape, self.get_output_shape()):
            print type(self), res.shape, self.get_output_shape()
            assert False
        return res
    return wrapper
# Relu Node

class ReluNode(ZeroArgNode):
    @shape_assertion
    def forward_pass(self, data, params):
        return np.maximum(data, np.zeros(data.shape))

# Dropout Node
USE_DROPOUT = True
class DropoutNode(ZeroArgNode):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
    @shape_assertion
    def forward_pass(self, data, params):
        if USE_DROPOUT:
            return np.random.binomial(1, 1.0-self.dropout_rate, data.shape) * data / (1.0-self.dropout_rate)
        else:
            return data

class MatrixMultiplyingNode(NNNode):
    def __init__(self, output_size):
        self.output_size = output_size
    def set_input_shape(self, shape):
        for i in range(len(shape)):
            if i != 1:
                assert abs(shape[i]) == 1
        self.input_shape = shape
        self.params_shape = (self.input_shape[1], self.output_size)
        self.output_shape = (self.input_shape[0], self.output_size)
    def get_output_shape(self):
        return self.output_shape
    @shape_assertion
    def forward_pass(self, data, params):
        M = params.get(self.params_shape)
        return np.dot(data, M)

class BiasNode(NNNode):
    def set_input_shape(self, shape):
        self.input_shape = shape
        self.params_shape = shape[1:]
    def get_output_shape(self):
        return self.input_shape
    @shape_assertion
    def forward_pass(self, data, params):
        B = params.get(self.params_shape)
        return data + B

class PerceptronLayer(NNetwork):
    def __init__(self, output_size):
        NNetwork.__init__(self)
        self.output_size = output_size
    def set_input_shape(self, input_shape):
        NNetwork.set_input_shape(self, input_shape)
        self.add_node(MatrixMultiplyingNode(self.output_size))
        self.add_node(BiasNode())

class NormalizationNode(NNNode):
    def set_input_shape(self, shape):
        self.input_shape = shape
    def get_output_shape(self):
        return self.input_shape
    def get_params_shape(self):
        return (2,)
    def forward_pass(self, data, params):
        mean, std = params.get(self.get_params_shape())
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        return (data * std) + mean

# Convolution i Maxpool odgapiony od implementacji w autogradzie
class MaxPoolNode(ZeroArgNode):
    def get_output_shape(self):
        return self.input_shape[:2] + (self.input_shape[2]/2, self.input_shape[3]/2)
    @shape_assertion
    def forward_pass(self, data, params):
        assert len(data.shape) == 4
        w, h = self.input_shape[2:]
        assert w%2==0 and h%2==0
        data = data.reshape(self.input_shape[:2] + (w/2, 2, h/2, 2))
        return np.max(np.max(data, axis=3), axis=4)

class ReshapeNode(ZeroArgNode):
    def __init__(self, shape):
        self.output_shape = shape
    def get_output_shape(self):
        return self.output_shape
    @shape_assertion
    def forward_pass(self, data, params):
        return data.reshape(self.output_shape)

class ConvolutionNode(NNNode):
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
    def set_input_shape(self, shape):
        self.input_shape = shape
        print shape
        assert len(shape) == 4
    def get_output_shape(self):
        return (self.input_shape[0], self.num_filters, self.input_shape[2] - self.filter_size + 1, self.input_shape[3] - self.filter_size + 1)
    def get_params_shape(self):
        return (self.input_shape[1], self.num_filters, self.filter_size, self.filter_size)
    @shape_assertion
    def forward_pass(self, data, params):
        filters = params.get(self.get_params_shape())
        return convolve(data, filters, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')

class LogsoftmaxNode(ZeroArgNode):
    def log_softmax(self, batch):
        batch = batch - np.max(batch, axis=1, keepdims=True)
        return batch - logsumexp(batch, axis=1).reshape((batch.shape[0], -1))
    @shape_assertion
    def forward_pass(self, data, params):
        return self.log_softmax(data)
