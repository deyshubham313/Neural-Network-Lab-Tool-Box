"""
modules/backward_prop.py
Two-hidden-layer MLP trained with mini-batch SGD + backprop.
Pure NumPy – no framework dependency.
"""
import numpy as np
import plotly.graph_objects as go


def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(x): return sigmoid(x) * (1 - sigmoid(x))
def relu(x):     return np.maximum(0, x)
def relu_d(x):   return (x > 0).astype(float)


def _generate_data(problem: str, n: int = 200):
    np.random.seed(42)
    if problem == "XOR":
        X = np.random.randn(n, 2)
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
    elif problem == "Circle":
        angles = np.random.uniform(0, 2*np.pi, n)
        r = np.random.uniform(0, 1, n)
        X = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        y = (r > 0.5).astype(float)
    elif problem == "Spiral":
        n2 = n // 2
        t  = np.linspace(0, 4, n2)
        X1 = np.column_stack([t * np.cos(t), t * np.sin(t)])
        X2 = np.column_stack([t * np.cos(t + np.pi), t * np.sin(t + np.pi)])
        X  = np.vstack([X1, X2]) + np.random.randn(n, 2) * 0.15
        y  = np.hstack([np.zeros(n2), np.ones(n2)])
    else:  # Moons
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n, noise=0.2, random_state=42)
        X = X.astype(float)
        y = y.astype(float)
    # Normalize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, y


class BackpropNetwork:
    def __init__(self, lr=0.05, h1=8, h2=4):
        self.lr = lr
        self.h1 = h1
        self.h2 = h2
        self.params = {}

    def _init_params(self, n_in=2):
        h1, h2 = self.h1, self.h2
        np.random.seed(1)
        self.params = {
            'W1': np.random.randn(h1, n_in)  * np.sqrt(2/n_in),
            'b1': np.zeros((h1, 1)),
            'W2': np.random.randn(h2, h1)    * np.sqrt(2/h1) if h2 else None,
            'b2': np.zeros((h2, 1))           if h2 else None,
            'W3': np.random.randn(1, h2 or h1) * np.sqrt(2/(h2 or h1)),
            'b3': np.zeros((1, 1)),
        }

    def _forward(self, X):
        p = self.params
        Z1 = p['W1'] @ X.T + p['b1']
        A1 = relu(Z1)
        if self.h2:
            Z2 = p['W2'] @ A1 + p['b2']
            A2 = relu(Z2)
        else:
            Z2, A2 = None, A1
        Z3 = p['W3'] @ (A2 if self.h2 else A1) + p['b3']
        A3 = sigmoid(Z3)
        return Z1, A1, Z2, A2, Z3, A3

    def _backward(self, X, y, Z1, A1, Z2, A2, Z3, A3):
        p  = self.params
        m  = X.shape[0]
        dZ3 = A3 - y.reshape(1, -1)
        dW3 = dZ3 @ (A2 if self.h2 else A1).T / m
        db3 = dZ3.mean(1, keepdims=True)
        grad_norm = np.linalg.norm(dW3)

        if self.h2:
            dA2 = p['W3'].T @ dZ3
            dZ2 = dA2 * relu_d(Z2)
            dW2 = dZ2 @ A1.T / m
            db2 = dZ2.mean(1, keepdims=True)
            dA1 = p['W2'].T @ dZ2
        else:
            dA1 = p['W3'].T @ dZ3

        dZ1 = dA1 * relu_d(Z1)
        dW1 = dZ1 @ X / m
        db1 = dZ1.mean(1, keepdims=True)

        p['W3'] -= self.lr * dW3
        p['b3'] -= self.lr * db3
        p['W1'] -= self.lr * dW1
        p['b1'] -= self.lr * db1
        if self.h2:
            p['W2'] -= self.lr * dW2
            p['b2'] -= self.lr * db2

        return grad_norm

    def train(self, problem="XOR", epochs=1000):
        self.X, self.y = _generate_data(problem)
        self._init_params()
        losses, grad_norms = [], []
        for _ in range(epochs):
            Z1, A1, Z2, A2, Z3, A3 = self._forward(self.X)
            loss = -np.mean(self.y * np.log(A3 + 1e-8) + (1 - self.y) * np.log(1 - A3 + 1e-8))
            gn   = self._backward(self.X, self.y, Z1, A1, Z2, A2, Z3, A3)
            losses.append(float(loss))
            grad_norms.append(float(gn))
        return losses, grad_norms

    def accuracy(self, problem):
        _, _, _, _, _, A3 = self._forward(self.X)
        preds = (A3.flatten() > 0.5).astype(int)
        return 100 * np.mean(preds == self.y.astype(int))

    def plot_decision_boundary(self, problem):
        xx = np.linspace(self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5, 120)
        yy = np.linspace(self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5, 120)
        XX, YY = np.meshgrid(xx, yy)
        grid = np.c_[XX.ravel(), YY.ravel()]
        _, _, _, _, _, A3 = self._forward(grid)
        Z = A3.reshape(XX.shape)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=xx, y=yy, z=Z,
            colorscale=[[0,'rgba(124,58,237,.4)'],[0.5,'rgba(10,21,32,.2)'],[1,'rgba(0,229,255,.4)']],
            showscale=False, contours=dict(showlines=False),
            hoverinfo='skip'))

        for cls, color, name in [(0,'#7c3aed','Class 0'), (1,'#00e5ff','Class 1')]:
            mask = self.y == cls
            fig.add_trace(go.Scatter(
                x=self.X[mask, 0], y=self.X[mask, 1],
                mode='markers', name=name,
                marker=dict(color=color, size=6,
                            line=dict(color='rgba(255,255,255,.3)', width=1))))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#5a7a9a', family='JetBrains Mono'),
            xaxis=dict(gridcolor='#1a3550', color='#5a7a9a', title='x₁'),
            yaxis=dict(gridcolor='#1a3550', color='#5a7a9a', title='x₂'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#5a7a9a')),
            margin=dict(l=10,r=10,t=10,b=10), height=320)
        return fig
