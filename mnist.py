# Import MNIST data module
from tensorflow.examples.tutorials.mnist import input_data

def get_MNIST(target_folder):
    # MNIST dataset will be downloaded if not located in the given path
    # one_hot=True means that the labels have been converted from a single
    # number to a vector whose length equals the number of possible classes
    return input_data.read_data_sets(target_folder, one_hot=True)

def print_sizes(data):
    # Print the sizes of the 3 mutually exclusive data sets
    print("Size of:")
    print("Training set:", len(data.train.labels))
    print("Test set:", len(data.test.labels))
    print("Validation set:", len(data.validation.labels))
