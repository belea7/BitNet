"""
File for class Neuron.
"""


class Neuron:
    """
    Represents a neuron in the Neural Network.
    """
    def __init__(self, weights, neighbors=[], output=False):
        """
        Constructor for class Neuron.

        :param weights: list of weights that corresponds to the neighbors and the inputs.
        :param neighbors: the neighboring neurons.
        :param output: is the neuron an output neuron.
        """
        self.weights = weights          # Weights of the input connections
        self.neighbors = neighbors      # The neighboring neurons
        self.output = output            # If the neuron is an output neuron

    def get_output(self, inputs):
        """
        Feed inputs to the neuron.
        Performs linear combination of the inputs with the weights.

        :param inputs: the inputs to feed.
        :returns: the output of the neuron.
        """
        output = self.linear_combination(inputs)
        if self.output:
            if output >= 0:
                return 1
            else:
                return -1
        return output

    def linear_combination(self, inputs):
        """
        Performs linear combination on inputs and weights.

        :returns: result of linear combination
        """
        sum = 0
        for index in range(len(inputs)):
            sum += (inputs[index] * self.weights[index])
        return sum
