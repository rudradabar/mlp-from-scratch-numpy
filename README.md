# MLP from Scratch (NumPy)

A small, educational multilayer perceptron built with NumPy. It learns the XOR problem using forward propagation, backpropagation, and gradient descent—without a deep-learning framework.

## Requirements

- Python 3.9 or newer
- NumPy

Install the dependency:

```bash
python -m pip install -r requirements.txt
```

## Run the example

```bash
python example.py
```

The example trains a network with two inputs, one hidden layer, and one sigmoid output. It prints the loss during training and the predicted probability for each XOR input.

## Use the network

```python
import numpy as np
from mlp import MLP

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

model = MLP(layer_sizes=[2, 8, 1], learning_rate=0.5, seed=7)
model.fit(X, y, epochs=10_000, print_every=1_000)
probabilities = model.predict(X)
predictions = (probabilities >= 0.5).astype(int)
```

`layer_sizes` lists the number of units in each layer, including the input and output layers. This implementation uses ReLU in hidden layers and sigmoid with binary cross-entropy in the output layer, so targets should be binary values shaped `(number_of_samples, 1)`.

## Project files

- `mlp.py` — network implementation and training loop
- `example.py` — runnable XOR demonstration
- `requirements.txt` — Python dependency

## How it works

Each layer computes `activation(previous_layer @ weights + bias)`. During training, binary cross-entropy measures prediction error. Backpropagation applies the chain rule to calculate gradients, and gradient descent updates the weights and biases.

This is a learning example rather than a production machine-learning library. It is designed for small binary classification problems.
