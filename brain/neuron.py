import numpy as np

class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.9):
        self.threshold = threshold
        self.decay = decay
        self.v = 0.0

    def step(self, current):
        self.v = self.decay * self.v + current
        if self.v >= self.threshold:
            self.v = 0.0
            return 1
        return 0
