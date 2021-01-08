"""
Contains Input Layer and Output Layer classes
"""
import constants as const
from neuron import Neuron


class InputLayer:
    """
    Represents the output layer of the network.
    """
    def __init__(self, bias):
        """
        Constructor for InputLayer class.
        :param bias: should bias neuron be added to the layer
        """
        self.bias = bias
        self.neurons = self.init_neurons()

    def init_neurons(self):
        """
        Creates the neurons in the layer.
        If bias=True a bias neuron is added.
        The weights are initialized as zeros.

        :returns: None
        """
        neurons = []
        for i in range(const.INPUT_LAYER_SIZE):
            new = Neuron(neighbors=[], weights=[const.INPUT_LAYER_WEIGHTS[i]])
            neurons.append(new)
        if self.bias:
            new = Neuron(neighbors=[], weights=[1])
            neurons.append(new)
        return neurons

    def get_outputs(self, inputs):
        """
        Feeds the input to the network and returns the output.

        :param inputs: the input of the network (list of binary digits).
        :returns: the output of the layer
        """
        outputs = []
        for i in range(len(inputs)):
            outputs.append(self.neurons[i].get_output([inputs[i]]))
        return outputs


class OutputLayer:
    """
    Represents the output layer of the network.
    """
    def __init__(self, inputLayer):
        """"
        Constructor for OutputLayer class.

        :param inputLayer: the input layer in the network.
        """
        self.inputLayer = inputLayer

        # Initialize weights as zeros
        weights = const.INIT_WEIGHTS
        if self.inputLayer.bias:
            weights.append(0)

        # Create output neuron
        self.neuron = Neuron(neighbors=inputLayer.neurons,
                             weights=weights,
                             output=True)

    def add_vector_to_weights(self, vector):
        """
        Changes the weights of a neuron by adding a vector to the weights vector.

        :param vector: vector to add to the weights
        :returns: None
        """
        for index in range(len(vector)):
            self.neuron.weights[index] += vector[index]

    def get_output(self, inputs):
        """
        Feeds the inputs to the input layer and then its output to the output layer.

        :param inputs: the inputs to the network.
        :returns: Returns the networks output.
        """
        output = self.inputLayer.get_outputs(inputs)
        return self.neuron.get_output(output)
