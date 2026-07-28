import numpy as np

#y = W @ x + b

# Input vector
x = np.array([10, 20])


# Weight matrix
W = np.array([
    [2, 0],
    [0, 3]
])


# Bias vector
b = np.array([1, 5])


# Neural network layer operation
y = W @ x + b


print("Input:")
print(x)

print("\nWeights:")
print(W)

print("\nBias:")
print(b)

print("\nOutput:")
# print(y)
