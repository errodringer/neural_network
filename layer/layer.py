import numpy as np

from neuron.neuron import Neuron


class Layer:
    """
    A layer of neurons in the neural network.

    Attributes:
        neurons (list): List of Neuron objects in the layer.
    """

    def __init__(self, num_neurons, input_size):
        """
        Initializes a layer with a specified number of neurons.

        Parameters:
            num_neurons (int): Number of neurons in the layer.
            input_size (int): Number of inputs to each neuron in the layer.
        """
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def forward(self, inputs):
        """
        Computes the forward pass for the layer.

        Parameters:
            inputs (ndarray): The input values to the layer.

        Returns:
            ndarray: The activated outputs of the neurons in the layer.
        """
        outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return outputs

    def backward(self, d_outputs, learning_rate):
        """
        Computes the backward pass and updates the neurons in the layer.

        Parameters:
            d_outputs (ndarray): The gradients of the loss with respect to the outputs of the layer.
            learning_rate (float): The learning rate for weight updates.

        Returns:
            ndarray: The gradients of the loss with respect to the inputs to the layer.
        """
        d_inputs = np.zeros(len(self.neurons[0].input))  # Initialize gradient wrt inputs
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_outputs[i], learning_rate)
        return d_inputs

    def to_dict(self):
        """
        Converts the layer's neurons to a dictionary format for saving.

        Returns:
            list: A list of dictionaries, each representing a neuron's parameters.
        """
        return [neuron.to_dict() for neuron in self.neurons]

    def from_dict(self, data):
        """
        Loads the layer's neurons from a dictionary.

        Parameters:
            data (list): A list of dictionaries, each representing a neuron's parameters.
        """
        for neuron, neuron_data in zip(self.neurons, data):
            neuron.from_dict(neuron_data)


# Example usage
if __name__ == "__main__":
    # Create a layer with 3 neurons, each receiving 4 inputs
    layer = Layer(3, 4)

    # Example inputs (4 features)
    inputs = np.array([1.0, 0.5, -1.5, 2.0])

    # Compute the output of the layer
    layer_output = layer.forward(inputs)

    print("Layer output:", layer_output)
