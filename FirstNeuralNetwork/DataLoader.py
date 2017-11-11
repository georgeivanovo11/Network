import matplotlib.image as mpimg
import numpy as np
import _pickle as cPickle
import gzip

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, val_data, test_data = cPickle.load(f,encoding='latin1')
    f.close()
    return (training_data, val_data, test_data)
    # train_data consists of 2 entries:
    # train_data[0] - array of images
    # train_data[0][0...49999] - array is composed of 50000 images
    # train_data[0][0][0...783] - each image(28*28px) is matrix of 784 values
    # train_data[1] - array of numbers
    # train_data[1][0...49999] - array is composed of 50000 numbers (0...9)

def load_train_data(number_of_rows):
    tr_d, rv_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0][:number_of_rows]]
    training_results = [vectorized_result(y) for y in tr_d[1][:number_of_rows]]
    training_data = zip(training_inputs, training_results)
    return list(training_data)
    #training_data is composed of "number_of_rows" elementa
    #each element is 2-tuples: the image and the digit
    #the image is 784-dimensional array with values from 0 to 1
    #the digit is 10-dimensional array representing the digit

def load_test_data(number_of_rows):
    tr_d, rv_d, te_d = load_data()
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0][:number_of_rows]]
    test_results =  te_d[1][:number_of_rows]
    test_data = zip(test_inputs, test_results)
    return list(test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_image(path):
    img = mpimg.imread(path)
    gray = rgb(img)
    array = []
    for row in gray:
        for item in row:
            c = []
            c.append(1 - item)
            array.append(c)
    return np.array(array)


def rgb(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])