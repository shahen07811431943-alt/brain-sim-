import numpy as np
from .neuron import LIFNeuron

class NeuralLayer:
    def __init__(self, n_in, n_out, seed=42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.5, (n_out, n_in))
        self.neurons = [LIFNeuron() for _ in range(n_out)]

    def forward(self, x):
        currents = self.W @ x
        return np.array([n.step(c) for n, c in zip(self.neurons, currents)], dtype=int)
