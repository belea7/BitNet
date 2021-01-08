"""
Contains Neural Network class.
"""
from layers import *
from data_generator import data_generator
from tabulate import tabulate
from termcolor import colored


class NeuralNetwork:
    """
    Class representing Neural Network
    """
    def __init__(self, bias):
        """
        Constructor for Neural Network class.

        :param bias: does the neural network contain a bias neuron.
        """
        self.inputLayer = InputLayer(bias)                  # Input layer
        self.outputLayer = OutputLayer(self.inputLayer)     # Output layer
        self.data = {}                                      # Data collected from networks training
        self.iteration = 1                                  # Training iteration
        self.bias = bias                                    # Does the neural network contain a bias neuron.

    def get_output(self, input):
        """
        Gets Neural Network output.
        The input is a binary number string.
        The output is: 1 - the input has more 1-s than 0-s.
                       2 - the input has more 0-s than 1-s.

        :param input: the input of the network.
        :return: the output
        """
        new_input = [1 if x == '1' else -1 for x in list(input)]
        if self.bias:
            new_input.append(1)
        return self.outputLayer.get_output(new_input)

    def fix_weights(self, example, output, expected):
        """
        Performs the following algorithm:
            1. For a certain example check is the output equals to the expected output.
            2. If the output is not as expected:
                2.1 If output >=0: multiply the input by (-1) and add to weights vector.
                2.2 else: add input to weights vector.
                3.3 Update data dict.
            3. Return if the example was misclassified.

        :returns: was the example misclassified
        """
        if output != expected:
            if output >= 0:
                vector = [-1 if x == '1' else 1 for x in list(example)]
                if self.bias:
                    vector.append(-1)
            else:
                vector = [1 if x == '1' else -1 for x in list(example)]
                if self.bias:
                    vector.append(1)
            self.outputLayer.add_vector_to_weights(vector)

            if expected == 1:
                if self.iteration not in self.data:
                    self.data[self.iteration] = {}
                    self.data[self.iteration]["g1_wrong"] = 1
                else:
                    if "g1_wrong" not in self.data[self.iteration]:
                        self.data[self.iteration]["g1_wrong"] = 1
                    else:
                        self.data[self.iteration]["g1_wrong"] += 1
            else:
                if self.iteration not in self.data:
                    self.data[self.iteration] = {}
                    self.data[self.iteration]["g2_wrong"] = 1
                else:
                    if "g2_wrong" not in self.data[self.iteration]:
                        self.data[self.iteration]["g2_wrong"] = 1
                    else:
                        self.data[self.iteration]["g2_wrong"] += 1
            return True
        return False

    def check_outputs(self, inputs, outputs, examples):
        """
        Checks the number of inputs that were classified wrong.
        Calculates and prints accuracy percentage.

        :param inputs: inputs fed to the network.
        :param outputs: the inputs given to the network.
        :param examples: the outputs to the inputs.
        :return: list of inputs classified wrong.
        """
        wrong = []
        for input in inputs:
            if examples[input] != outputs[input]:
                wrong.append(input)
        accuracy = 1 - (len(wrong) / len(inputs))
        color = "green" if accuracy >= 0.8 else "red"
        print("Accuracy percentage: " + colored("{0:.0%}".format(accuracy), color))
        return wrong

    def train(self, examples):
        """
        Train the neural network using perceptron training algorithm.
        Performs the following algorithm:
            1. While not all examples are correctly classified:
                1.1 For every example:
                    1.1.1 Feed to the network.
                    1.1.2 If misclassified - update the weights.
                    1.1.3 Update data dict.
                    1.1.4 Increase iteration count.

        :param examples: examples of inputs with corresponding outputs.
        :returns: None
        """
        inputs = list(examples.keys())
        errors = inputs.copy()
        while errors:
            errors = []
            # Feed inputs to network
            outputs = {}
            for i in inputs:
                outputs[i] = self.get_output(i)
                wrong = self.fix_weights(i, outputs[i], examples[i])
                if wrong:
                    errors.append(i)
            if self.iteration not in self.data:
                self.data[self.iteration] = {}
            self.data[self.iteration]["total"] = len(errors)
            self.iteration += 1

    def test(self, examples):
        """
        Run network on testing set and check its accuracy.

        :param examples: examples of inputs with corresponding outputs.
        :returns: None
        """
        inputs = list(examples.keys())
        outputs = {}
        for input in inputs:
            outputs[input] = self.get_output(input)

        wrong = self.check_outputs(inputs, outputs, examples)
        self.print_testing_table(wrong, examples)
        print(self.outputLayer.neuron.weights)

    def print_training_table(self):
        """
        Prints table that describes networks behavior during training phase.
        For each iteration displays number of examples misclassified from each group.

        :returns: None
        """
        rows = []
        for cycle in self.data.keys():
            data = self.data[cycle]
            if "g1_wrong" not in data:
                data["g1_wrong"] = 0
            if "g2_wrong" not in data:
                data["g2_wrong"] = 0
            row = ["Iteration #{}".format(cycle), colored(data["g1_wrong"], "blue"),
                   colored(data["g2_wrong"], "red"), data["total"]]
            rows.append(row)

        headers = ["Iteration num.", colored("More 1's wrong", "blue"), colored("More 0's wrong", "red"), "Total wrong"]
        table = tabulate(tabular_data=rows, headers=headers, tablefmt='orgtbl', numalign="center")
        print("\n" + table + "\n")

    def print_testing_table(self, wrong, examples):
        """
        Prints a table that summarizes the results of the testing phase.
        Displays for each group: number of examples misclassified, number of examples classified correctly.

        :returns: None
        """
        more_ones_correct = 0
        more_ones_wrong = 0
        more_zeros_correct = 0
        more_zeros_wrong = 0
        for number in examples.keys():
            if examples[number] == 1:
                if number in wrong:
                    more_ones_wrong += 1
                else:
                    more_ones_correct += 1
            else:
                if number in wrong:
                    more_zeros_wrong += 1
                else:
                    more_zeros_correct += 1

        row1 = [colored("More 1's", "blue"),
                colored(more_ones_correct, "blue"), colored(more_ones_wrong, "blue")]
        row2 = [colored("More 0's", "red"),
                colored(more_zeros_correct, "red"), colored(more_zeros_wrong, "red")]
        row3 = ["Total", more_ones_correct + more_zeros_correct, more_ones_wrong + more_zeros_wrong]
        headers = ["Group", "Correct", "Wrong"]
        table = tabulate(tabular_data=[row1, row2, row3], headers=headers, tablefmt='orgtbl', numalign="center")
        print("\nTesting results:")
        print(table + "\n")


if __name__ == '__main__':
    print("Creating examples...")
    train_set, test_set = data_generator()
    network = NeuralNetwork(bias=const.BIAS)
    print("Training the network...")
    network.train(train_set)
    network.print_training_table()
    print("Network is trained (after {} iterations), testing the network.".format(network.iteration))
    network.test(test_set)
