"""
modules/perceptron.py
Single-layer Perceptron with perceptron learning rule.
"""
import numpy as np


GATE_TARGETS = {
    "AND":  [0, 0, 0, 1],
    "OR":   [0, 1, 1, 1],
    "NAND": [1, 1, 1, 0],
    "NOR":  [1, 0, 0, 0],
    "XOR":  [0, 1, 1, 0],
}

INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)


def step(z):
    return 1 if z >= 0 else 0


class Perceptron:
    def __init__(self, lr: float = 0.1):
        self.lr = lr
        self.w  = np.random.uniform(-0.5, 0.5, 2)
        self.b  = np.random.uniform(-0.5, 0.5)

    def predict_one(self, x):
        return step(np.dot(self.w, x) + self.b)

    def get_targets(self, gate: str):
        return GATE_TARGETS[gate]

    def train(self, gate: str = "AND", epochs: int = 50):
        targets = GATE_TARGETS[gate]
        losses  = []
        weights_history = []

        for _ in range(epochs):
            epoch_loss = 0
            for x, y in zip(INPUTS, targets):
                pred = self.predict_one(x)
                err  = y - pred
                self.w += self.lr * err * x
                self.b += self.lr * err
                epoch_loss += abs(err)
            losses.append(epoch_loss)
            weights_history.append((self.w.copy(), self.b))

        return losses, weights_history

    def evaluate(self, gate: str):
        targets = GATE_TARGETS[gate]
        preds   = [self.predict_one(x) for x in INPUTS]
        correct = sum(p == t for p, t in zip(preds, targets))
        return preds, correct
