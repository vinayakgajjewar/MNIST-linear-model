from tensorflow.examples.tutorials.mnist import input_data

def get_MNIST(target_folder):
    return input_data.read_data_sets(target_folder, one_hot=True)

def print_sizes(data):
    print("Size of:")
    print("Training set:", len(data.train.labels))
    print("Test set:", len(data.test.labels))
    print("Validation set:", len(data.validation.labels))
