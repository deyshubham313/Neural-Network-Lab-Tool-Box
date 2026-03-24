"""
modules/forward_prop.py
Multi-layer feedforward network – pure NumPy, forward pass only.
"""
import numpy as np


def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh(x):    return np.tanh(x)
def linear(x):  return x

ACTS = {"ReLU": relu, "Sigmoid": sigmoid, "Tanh": tanh, "Linear": linear}


class ForwardPropNetwork:
    def __init__(self, n_in, n_h1, n_h2, n_out, activation="ReLU"):
        self.act_fn = ACTS.get(activation, relu)
        self.layers = []

        # Input → H1
        prev = n_in
        if n_h1 > 0:
            self.layers.append((np.random.randn(n_h1, prev) * 0.5,
                                 np.zeros((n_h1, 1))))
            prev = n_h1
        # H1 → H2
        if n_h2 > 0:
            self.layers.append((np.random.randn(n_h2, prev) * 0.5,
                                 np.zeros((n_h2, 1))))
            prev = n_h2
        # → Output  (sigmoid on output)
        self.layers.append((np.random.randn(n_out, prev) * 0.5,
                             np.zeros((n_out, 1))))

    def forward(self, x: np.ndarray):
        a = x.reshape(-1, 1)
        activations = [a]
        for i, (W, b) in enumerate(self.layers):
            z = W @ a + b
            a = sigmoid(z) if i == len(self.layers) - 1 else self.act_fn(z)
            activations.append(a)
        return activations
