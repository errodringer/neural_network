import numpy as np


class Neuron:
    """
    A single neuron in the neural network.

    Attributes:
        weights (ndarray): Weights associated with the inputs.
        bias (float): Bias term added to the neuron's weighted sum.
        output (float): Output of the neuron after activation.
        input (ndarray): Inputs to the neuron during the forward pass.
        dweights (ndarray): Gradient of the loss with respect to the weights.
        dbias (float): Gradient of the loss with respect to the bias.
    """

    def __init__(self, input_size):
        """
        Initializes a Neuron with random weights and bias.

        Parameters:
            input_size (int): The number of inputs to the neuron.
        """
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.output = 0
        self.input = None
        self.dweights = np.zeros_like(self.weights)
        self.dbias = 0

    def activate(self, x):
        """
        Applies the sigmoid activation function.

        Parameters:
            x (float): The input value.

        Returns:
            float: Activated output value.
        """
        return 1 / (1 + np.exp(-x))

    def activate_derivative(self, x):
        """
        Computes the derivative of the sigmoid function.

        Parameters:
            x (float): The activated output value.

        Returns:
            float: Derivative of the sigmoid.
        """
        return x * (1 - x)

    def forward(self, inputs):
        """
        Computes the forward pass for the neuron.

        Parameters:
            inputs (ndarray): The input values to the neuron.

        Returns:
            float: The activated output of the neuron.
        """
        self.input = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output

    def backward(self, d_output, learning_rate):
        """
        Computes the backward pass and updates weights and bias.

        Parameters:
            d_output (float): The gradient of the loss with respect to the output.
            learning_rate (float): The learning rate for weight updates.

        Returns:
            ndarray: The gradient of the loss with respect to the input.
        """
        d_activation = d_output * self.activate_derivative(self.output)
        self.dweights = np.dot(self.input, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weights)
        # Update weights and biases
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
        return d_input

    def to_dict(self):
        """
        Converts the neuron's weights and bias to a dictionary format for saving.

        Returns:
            dict: A dictionary containing the weights and bias of the neuron.
        """
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias
        }

    def from_dict(self, data):
        """
        Loads the neuron's weights and bias from a dictionary.

        Parameters:
            data (dict): A dictionary containing the weights and bias to load.
        """
        self.weights = np.array(data["weights"])
        self.bias = data["bias"]


# Example usage
if __name__ == "__main__":
    # Create a neuron with 3 inputs
    neuron = Neuron(3)

    # Example inputs
    inputs = np.array([1.5, 2.0, -1.0])

    # Compute the neuron's output
    output = neuron.forward(inputs)

    print("Neuron output:", output)
