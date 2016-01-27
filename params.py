import autograd.numpy as np
from operator import mul

class Params:
    def __init__(self):
        self.w = []
        self.ptr = 0
    def feed(self, w):
        if self.ptr != len(self.w):
            raise IndexError("previous vector not exhausted")
        self.w = w
        self.ptr = 0
    def get(self, shape):
        length = abs(reduce(mul, shape))
        if self.ptr + length > len(self.w):
            raise IndexError("wtf")
        self.ptr += length
        return self.w[self.ptr-length:self.ptr].reshape((shape))
