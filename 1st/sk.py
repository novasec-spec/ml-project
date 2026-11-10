# mnist.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from nn import NeuralNet, cross_entropy

print("Downloading MNIST (first run only, ~15 MB)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# One-hot
Y = np.eye(10)[y]

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y, test_size=0.2, random_state=0
)

# Use a subset first so pure NumPy isn't painfully slow
N = 8000
X_train, Y_train, y_train = X_train[:N], Y_train[:N], y_train[:N]

net = NeuralNet([784, 128, 10], hidden_act="relu")

EPOCHS = 60
LR = 0.2
for epoch in range(EPOCHS):
    net.forward(X_train)
    net.backward(X_train, Y_train, LR)
    if epoch % 5 == 0:
        loss = cross_entropy(net.activations[-1], Y_train)
        train_acc = net.accuracy(X_train, y_train)
        print(f"epoch {epoch:3d}  loss={loss:.4f}  train_acc={train_acc:.4f}")

print(f"\nTest accuracy: {net.accuracy(X_test, y_test):.4f}")
