# MLP from Scratch (NumPy)

A beginner-friendly multilayer perceptron that learns the XOR problem using NumPy. It demonstrates forward propagation, binary cross-entropy, backpropagation, and gradient descent without a deep-learning framework.

## Requirements

- Python 3.9 or newer
- NumPy

Install NumPy:

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 mlp_from_scratch.py
```

The script trains a network with two inputs, one hidden layer of four `Tanh` neurons, and one `Sigmoid` output. It prints the loss during training and each XOR input's predicted probability and class.

## How it works

The forward pass sends each input through the dense layers and activation functions to produce a probability. Binary cross-entropy measures the error against the correct label. Backpropagation applies the chain rule to calculate gradients for the weights and biases, and gradient descent updates them to reduce the error over repeated training steps.

The hidden layer lets the network learn XOR, a pattern that a single perceptron cannot classify correctly.

## Project files

- `mlp_from_scratch.py` — network implementation, training loop, and runnable XOR example
- `requirements.txt` — NumPy dependency

This is an educational example for small binary classification problems, not a production machine-learning library.
