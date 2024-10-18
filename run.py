import numpy as np
import matplotlib.pyplot as plt

from neural_network.neural_network import NeuralNetwork


# Generate synthetic data for training
def generate_complex_data(num_samples=500, num_columns=3):
    """
    Generates a complex dataset with non-linear relationships between input features and output.

    Parameters:
        num_samples (int): The number of data samples to generate.
        num_columns (int): The number of data columns.

    Returns:
        X (ndarray): The generated input data with shape (num_samples, 3).
        y (ndarray): The generated target values with shape (num_samples, 1).
    """
    # Random input features
    X = np.random.rand(num_samples, num_columns) * 10  # Inputs in range [0, 10]

    # Create a "classification-like" target output between 0 and 1
    # Use a non-linear combination of input features and apply a sigmoid-like transformation
    logits = (0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2] ** 2 + np.random.randn(num_samples) * 0.1)

    # Apply sigmoid function to get values between 0 and 1
    y = 1 / (1 + np.exp(-logits))

    # Reshape y to be compatible with the neural network
    y = y.T

    return X, y


# Dummy training data (X: input, y: target)
X = np.array([[0.5, 0.2, 0.1],
              [0.9, 0.7, 0.3],
              [0.4, 0.5, 0.8]])

y = np.array([[0.3], [0.6], [0.9]])
y = np.array([[0.3, 0.6, 0.9]]).T

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
y = np.array([[0, 1, 1, 0]]).T

# X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# y = np.array([[0, 1, 1, 0]]).T
# y = np.array([[0], [1], [1], [0]])

# X, y = generate_complex_data()
# X_train = X[:400]
# y_train = y[:400]
# X_test = X[400:]
# y_test = y[400:]


nn = NeuralNetwork()

# Add layers: input size for first hidden layer is 3 (e.g., 3 input features)
nn.add_layer(num_neurons=3, input_size=2)  # First hidden layer with 5 neurons
nn.add_layer(num_neurons=3, input_size=3)  # Second hidden layer with 4 neurons
nn.add_layer(num_neurons=1, input_size=3)  # Output layer with 1 neuron

# Train the network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Predict using the trained network
y_pred = nn.predict(X)
# print(f"Predictions: {y_pred}")
# print(f"Real: {y_test}")

plt.plot(nn.loss_list)
plt.show()

plt.plot(y[:50])
plt.plot(y_pred[:50])
plt.show()