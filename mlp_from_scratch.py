"""MLP from Scratch (NumPy)

Beginner-friendly implementation of the MLP project in Neural Network
Foundations: Pseudocode for 5 Projects. It learns the XOR data with a
configurable stack of tanh hidden layers and a sigmoid output layer.

Run with: python3 mlp_from_scratch.py
Requires: numpy
"""

import numpy as np


class Dense:
    """A fully connected layer with trainable weights and biases."""

    def __init__(self, input_size, output_size, rng):
        # Small random weights break symmetry; zero biases are a simple start.
        self.weights = rng.normal(0.0, 0.5, (input_size, output_size))
        self.bias = np.zeros((1, output_size))
        self.input = None

    def forward(self, values):
        self.input = values
        return values @ self.weights + self.bias


class Tanh:
    """Hidden-layer activation that squashes values into [-1, 1]."""

    def forward(self, values):
        self.output = np.tanh(values)
        return self.output


class Sigmoid:
    """Output activation that turns a score into a probability in [0, 1]."""

    def forward(self, values):
        # Clipping prevents overflow in exp for unusually large scores.
        values = np.clip(values, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-values))
        return self.output


class MLP:
    """Multi-layer perceptron with tanh hidden layers and sigmoid output."""

    def __init__(self, input_size, hidden_sizes, output_size=1, seed=7):
        rng = np.random.default_rng(seed)
        self.layers = []
        previous_size = input_size

        # Build each hidden Dense -> Tanh pair in the requested order.
        for hidden_size in hidden_sizes:
            self.layers.append(Dense(previous_size, hidden_size, rng))
            self.layers.append(Tanh())
            previous_size = hidden_size

        # A sigmoid output is suitable for binary classification.
        self.layers.append(Dense(previous_size, output_size, rng))
        self.layers.append(Sigmoid())

    def forward(self, values):
        """Forward pass: carry the input through every layer to get a prediction."""
        output = values
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, prediction, target, learning_rate):
        """Backpropagate binary cross-entropy gradients and update parameters."""
        count = target.shape[0]

        # For sigmoid + binary cross-entropy, this is the gradient at the
        # output logits. Averaging by count makes the step size batch-size aware.
        gradient = (prediction - target) / count

        # Walk backwards: activation derivatives first, then Dense gradients.
        for layer in reversed(self.layers):
            if isinstance(layer, Sigmoid):
                gradient = gradient * layer.output * (1.0 - layer.output)
            elif isinstance(layer, Tanh):
                gradient = gradient * (1.0 - layer.output**2)
            elif isinstance(layer, Dense):
                weight_gradient = layer.input.T @ gradient
                bias_gradient = np.sum(gradient, axis=0, keepdims=True)
                gradient = gradient @ layer.weights.T

                # Gradient descent nudges each parameter against its gradient.
                layer.weights -= learning_rate * weight_gradient
                layer.bias -= learning_rate * bias_gradient

    def predict(self, values):
        """Return class labels (0 or 1) from the output probabilities."""
        return (self.forward(values) >= 0.5).astype(int)


def binary_cross_entropy(prediction, target):
    """Average binary cross-entropy loss; clipping avoids log(0)."""
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    return -np.mean(
        target * np.log(prediction) + (1.0 - target) * np.log(1.0 - prediction)
    )


def train(model, inputs, targets, epochs=10_000, learning_rate=0.1):
    """Train on the full dataset repeatedly and return the loss history."""
    losses = []
    for epoch in range(epochs):
        # Forward pass computes current predictions and the loss to minimize.
        prediction = model.forward(inputs)
        loss = binary_cross_entropy(prediction, targets)
        losses.append(loss)

        # Backpropagation computes gradients; gradient descent applies updates.
        model.backward(prediction, targets, learning_rate)

        if epoch % 2_000 == 0:
            print(f"epoch {epoch:5d} | loss {loss:.4f}")
    return losses


def main():
    # XOR truth table: neither a single perceptron nor a linear boundary can
    # separate these labels; a hidden layer lets the MLP learn the pattern.
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    model = MLP(input_size=2, hidden_sizes=[4], output_size=1, seed=7)
    train(model, X, y, epochs=10_000, learning_rate=0.1)

    probabilities = model.forward(X)
    labels = model.predict(X)
    print("\nXOR predictions:")
    for row, probability, label, expected in zip(X, probabilities, labels, y):
        print(
            f"{row.astype(int).tolist()} -> probability {probability[0]:.3f}, "
            f"class {label[0]} (expected {int(expected[0])})"
        )


if __name__ == "__main__":
    main()
