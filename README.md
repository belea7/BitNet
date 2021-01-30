Neural network tha classifies binary numbers by the number 1 bits in it
-------
**The program implements the neural network entirely (without using external modules).**

**Python modules: numpy, tabulate, termcolor.**

**Biological Computation course at The Open University of Israel**

I implemented a Perceptron neural network that solves the following classification problem: given a binary number containing 21 digets, does it contain more 1's than 0's or more 0's than 1's.
This is a single-layered Perceptron network, that contains has 22 neurons (21 input neurons and a bias neuron) in the input layer and a single output neuron.
It dynamically creates a train set and a test set (each containing 1000 classified examples) and uses the Perceptron on-line training algorithm to learng its weights.
The program prints informative messages:
- number of exmples from each category in the train and test sets
- how many examples from each category were missclassified in every iteration of the training algorithm
- the success rates of the nework on the test set

