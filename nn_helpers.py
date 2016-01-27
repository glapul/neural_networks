from params import Params
import nn_base 
import nn_types
import nn_helpers
import autograd.numpy as np
from autograd import grad
import random

BATCH_SIZE = 50
INIT_SCALE = 0.05
LEARNING_RATE = 0.01

def create_functions(network):
    def compute_inside(weights, data, dropout=True):
        params = Params()
        params.feed(weights)
        nn_types.USE_DROPOUT = dropout
        return network.forward_pass(data, params)
    def cost_f(weights, data, labels_hot_vectors):
        return -np.sum(compute_inside(weights, data) * labels_hot_vectors) / data.shape[0]
    def validating_f(weights, data, labels):
        return np.mean(np.argmax(labels, axis=1) != np.argmax(compute_inside(weights, data, False), axis=1))
    def classifying_f(weights, data):
        return np.argmax(compute_inside(weights, data, False), axis=1)
    return cost_f, validating_f, classifying_f

def create_batches(images, labels):
    l = zip(images, labels)
    random.shuffle(l)
    r = [l[i:i+BATCH_SIZE] for i in xrange(0, len(l), BATCH_SIZE)]
    res = []
    for i in r:
        images = []
        labels = []
        for j in i:
            images.append(j[0])
            labels.append(j[1])
        res.append((np.array(images), np.array(labels)))
    return res

def create_random_noise_array(shape):
    return INIT_SCALE * 2 * (np.random.random(shape) - 0.5)

def train_on_batch(weights, data, labels, cost_f):
    return weights - grad(cost_f)(weights, data, labels) * LEARNING_RATE
