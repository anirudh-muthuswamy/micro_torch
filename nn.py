from micrograd import Value
import numpy as np
import random
from functools import reduce

class Neuron:

    def __init__(self, nin):
        self.w = np.array([Value(random.uniform(-1, 1)) for _ in range(nin)], dtype=object)
        self.b = Value(random.uniform(-1, 1))

    def parameters(self):
        return np.append(self.w, self.b)

    def __call__(self, x):
        #w.x + b
        x = np.array(x, dtype = object)
        act = np.dot(self.w, x) + self.b
        out = act.tanh()
        return out

class Layer:

    def __init__(self, nin, nout):
        self.neurons = np.array([Neuron(nin) for _ in range(nout)], dtype=object)

    def parameters(self):
        return np.concatenate([neuron.parameters() for neuron in self.neurons])

    def __call__(self, x):
        x = np.array(x, dtype=object)
        outs = np.vectorize(lambda neuron: neuron(x))(self.neurons)
        return outs[0] if len(outs) == 1 else outs
    
class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = np.array([Layer(sz[i], sz[i+1]) for i in range(len(nouts))], dtype=object)

    def parameters(self):
        return np.concatenate([layer.parameters() for layer in self.layers])

    def __call__(self, x):
        return reduce(lambda acc, layer: layer(acc), self.layers, x)