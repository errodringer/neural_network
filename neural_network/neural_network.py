import json
import numpy as np
from layer.layer import Layer


class NeuralNetwork:
    """
    A simple feedforward neural network.

    Attributes:
        layers (list): List of Layer objects in the neural network.
    """

    def __init__(self):
        """
        Initializes an empty neural network with no layers.
        """
        self.layers = []
        self.loss_list = []

    def add_layer(self, num_neurons, input_size):
        """
        Adds a layer to the neural network.

        Parameters:
            num_neurons (int): The number of neurons in the new layer.
            input_size (int): The number of inputs to the new layer (or neurons).
        """
        if not self.layers:
            self.layers.append(Layer(num_neurons, input_size))
        else:
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size))

    def forward(self, inputs):
        """
        Computes the forward pass through the entire network.

        Parameters:
            inputs (ndarray): The input values to the network.

        Returns:
            ndarray: The output of the network after the forward pass.
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, learning_rate):
        """
        Computes the backward pass through the entire network.

        Parameters:
            loss_gradient (ndarray): The gradient of the loss with respect to the output of the network.
            learning_rate (float): The learning rate for weight updates.
        """
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate)

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """
        Trains the neural network using the given training data.

        Parameters:
            X (ndarray): The input training data (features).
            y (ndarray): The target output values (labels).
            epochs (int): The number of training iterations.
            learning_rate (float): The step size for weight updates.
        """
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                # Forward pass
                output = self.forward(X[i])
                # Calculate loss (mean squared error)
                loss += np.mean((y[i] - output) ** 2)
                # Backward pass (gradient of loss wrt output)
                loss_gradient = 2 * (output - y[i])
                self.backward(loss_gradient, learning_rate)
            loss /= len(X)
            self.loss_list.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Generates predictions for the input data.

        Parameters:
            X (ndarray): The input data for prediction.

        Returns:
            ndarray: The predicted outputs from the network.
        """
        predictions = []
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        return np.array(predictions)

    def save(self, filename="model.json"):
        """
        Saves the neural network's layers, neurons, weights, and biases to a JSON file.

        Parameters:
            filename (str): The name of the file to save the model parameters.
        """
        # Initialize the model data dictionary
        model_data = {
            "layers": []
        }

        # Iterate through each layer and store its configuration and neurons
        for layer in self.layers:
            layer_data = {
                "num_neurons": len(layer.neurons),
                "input_size": len(layer.neurons[0].weights) if layer.neurons else 0,
                "neurons": layer.to_dict()
            }
            model_data["layers"].append(layer_data)

        # Save the model data to a file
        with open(filename, "w") as f:
            json.dump(model_data, f)
        print(f"Model saved to {filename}")

    def load(self, filename="model.json"):
        """
        Loads the neural network's layers, neurons, weights, and biases from a JSON file.

        Parameters:
            filename (str): The name of the file from which to load the model parameters.
        """
        with open(filename, "r") as f:
            model_data = json.load(f)

        # Clear any existing layers before loading
        self.layers = []

        # Recreate the architecture based on the saved configuration
        for layer_data in model_data["layers"]:
            num_neurons = layer_data["num_neurons"]
            input_size = layer_data["input_size"]

            # Create a new layer with the specified number of neurons and input size
            new_layer = Layer(num_neurons, input_size)

            # Load neuron parameters from the saved data
            new_layer.from_dict(layer_data["neurons"])

            # Append the recreated layer to the network
            self.layers.append(new_layer)

        print(f"Model loaded from {filename}")


if __name__ == "__main__":
    # Example of usage:
    # Create a neural network with 3 input features, 1 hidden layers, and 1 output layer
    nn = NeuralNetwork()

    # Add layers: input size for first hidden layer is 3 (e.g., 3 input features)
    nn.add_layer(num_neurons=3, input_size=3)  # First hidden layer with 5 neurons
    nn.add_layer(num_neurons=3, input_size=3)  # Second hidden layer with 4 neurons
    nn.add_layer(num_neurons=1, input_size=4)  # Output layer with 1 neuron

    # Dummy training data (X: input, y: target)
    X = np.array([[0.5, 0.2, 0.1],
                  [0.9, 0.7, 0.3],
                  [0.4, 0.5, 0.8]])
    y = np.array([[0.3, 0.6, 0.9]]).T

    # Train the network
    nn.train(X, y, epochs=3000, learning_rate=0.5)

    # Save the model after training
    nn.save("../models/model.json")

    # Load the model (no need to train)
    nn.load("../models/model.json")

    # Predict using the trained network
    predictions = nn.predict(X)
    print(f"Predictions: {predictions}")
