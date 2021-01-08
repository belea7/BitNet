"""
Handles testing set and data set creation.
"""
from random import shuffle
import constants as const
from tabulate import tabulate
from termcolor import colored


def convert_to_binary(x):
    """
    Converts a number to binary base.

    :param x: number to convert
    :return: binary representation of a number
    """
    s = ""
    if x == 0:
        return "0"
    while x != 0:
        # add char to front of s
        s = str(x % 2) + s

        # integer division gives quotient
        x = int(x / 2)
    return s


def count(number):
    """
    Checks if the string has more 0's than 1's.
    If it does - return 1. Else - -1.

    :param number: the number to check
    :returns: 1 if the number has more 1's, else -1
    """
    s = sum([1 if x == '1' else -1 for x in list(number)])
    if s >= 0:
        return 1
    else:
        return -1


def add_zeros(number):
    """
    Adds zeros to the number.
    :param number: the number to add zeros to
    :returns: the number padded with zeros
    """
    return "0" * (21 - len(number)) + number


def print_table(data, train=True):
    """
    Print a table that contains information the data created.

    :param data: the data created (as train or test set).
    :param train: is the data a training set.
    :returns: None
    """
    # Find the number of examples in each group
    more_ones = []
    more_zeros = []
    for key in data.keys():
        if data[key] == 1:
            more_ones.append(key)
        else:
            more_zeros.append(key)

    # Print table
    row1 = [colored("More 1's", "blue"), colored(len(more_ones), "blue")]
    row2 = [colored("More 0's", "red"), colored(len(more_zeros), "red")]
    row3 = ["Total", len(more_ones) + len(more_zeros)]
    headers = ["Group", "Examples in {} set".format("train" if train else "test")]
    table = tabulate(tabular_data=[row1, row2, row3], headers=headers, tablefmt='orgtbl', numalign="center")
    print("\n" + table + "\n")


def data_generator():
    """
    Creates training set and test set by generating all binary numbers
    that are represented by 21 binary digits.
    Saves the training and testing sets to files.
    Prints informative table about the data created.

    :returns: data set and test set
    """
    max_number = 2097151    # 2097151=111111111111111111111
    numbers = []

    # Create all binary numbers
    for i in range(1, max_number + 1):
        binary = convert_to_binary(i)
        string = "{}".format(binary[0:])
        numbers.append(string)

    # Shuffle the numbers
    shuffle(numbers)

    # Create train set and save to file
    print("Creating training set ({} examples)".format(const.TRAIN_SET_SIZE))
    train_set = {}
    f = open(const.TRAIN_SET_FILE_NAME, "w+")
    for number in numbers[:const.TRAIN_SET_SIZE]:
        temp = add_zeros(number)
        train_set[temp] = count(temp)
        f.write("{0}\t{1}\n".format(temp, train_set[temp]))
    f.close()
    print("Training set is copied to {0}".format(const.TRAIN_SET_FILE_NAME))
    print_table(train_set)

    # Create test set and save to file
    print("Creating test set ({} examples)".format(const.TRAIN_SET_SIZE))
    test_set = {}
    f = open(const.TEST_SET_FILE_NAME, "w+")
    counter = 0
    for number in numbers[const.TRAIN_SET_SIZE:]:
        if counter == const.TEST_SET_SIZE:
            break
        temp = add_zeros(number)
        test_set[temp] = count(temp)
        f.write("{0}\t{1}\n".format(temp, test_set[temp]))
        counter += 1
    f.close()
    print("Test set is copied to {0}".format(const.TEST_SET_FILE_NAME))
    print_table(test_set, train=False)

    return train_set, test_set
