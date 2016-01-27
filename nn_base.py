import autograd.numpy as np
from operator import mul

class NNNode:
    def set_input_shape(self, shape):
        raise NotImplementedError()
    def get_output_shape(self):
        raise NotImplementedError()
    def get_params_shape(self):
        return self.params_shape
    def forward_pass(self, data, params):
        raise NotImplementedError()

class ZeroArgNode(NNNode):
    def set_input_shape(self, shape):
        self.input_shape = shape
    def get_output_shape(self):
        return self.input_shape
    def get_params_shape(self):
        return (0,)
    def forward_pass(self, data, params):
        raise NotImplementedError()

class NNetwork(NNNode):
    def __init__(self):
        self.nodes = []
        self.params_shape = (0,)
    def set_input_shape(self, input_shape):
        self.input_shape = input_shape 
    def add_node(self, node):
        if not self.nodes:
            node.set_input_shape(self.input_shape)
        else:
            node.set_input_shape(self.nodes[-1].get_output_shape())
        self.params_shape = (self.params_shape[0]+reduce(mul, node.get_params_shape()),)
        self.nodes.append(node)
    def get_output_shape(self):
        return self.nodes[-1].get_output_shape()
    def get_params_shape(self):
        return (self.params_shape)
    def forward_pass(self, data, params):
        for node in self.nodes:
            data = node.forward_pass(data, params)
        return data






