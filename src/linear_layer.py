import numpy as np


class LinearLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros(output_size)

    def forward(self, x):
        # Apply linear transformation
        return self.weights @ x + self.bias


# Create input vector
x = np.array([10, 20])


# Create a layer
layer = LinearLayer(input_size=2, output_size=3)


# Pass data through the layer
output = layer.forward(x)


print("Input:")
print(x)

print("\nWeights:")
print(layer.weights)

print("\nBias:")
print(layer.bias)

print("\nOutput:")
print(output)