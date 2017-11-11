import random
import numpy as np

class Network:

    numLayers = 0  # the number of layers
    sizes = [2,3,1]  # the number of neurons in the respective layers

    biases = [ [(0.2),              # array of biases of neurons in each layer except input-layer
                (0.3),              # hidden-layer has 3 neuron, output-layer has 1
                (0.1) ],
               [(0.2)]
             ]

    weights = [ [ (2,3,4),          #array of weights of neurons in each layer except output-layer
                  (3,5,6)],         #each of 2 input-neurons has synapses to each of 3 hidden-neurons
                [ (3),              #each of 3 hidden-neurons has synapse to 1 output-neuron
                  (4),
                  (5) ]
              ]

    def __init__(self, sizes):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.biases = [ np.random.randn(y,1) for y in sizes[1:] ]
        self.weights = [ np.random.randn(y,x)
                         for x,y in zip(sizes[:-1],sizes[1:])]

    #Note:
    #np.random.randn(A,B) - creates array with A rows and B columns, filled with random values
    #zip([2,3], [3,1]) -> [(2,3), (3,1)]
    #weihts[temp_layer][next_layer_neuron][temp_layer_neuron]


    def feedforward(self, a):
        for b,w in zip (self.biases, self.weights):
            a = (self.sigmoid(np.dot(w,a)+b))
        return a

    def SGM(self, trainData, epochs, miniBatchSize, learnRate):
        n = len(trainData)
        for j in range(0, epochs):
            random.shuffle(trainData)
            miniBatches = [trainData[k:k+miniBatchSize]
                           for k in range(0,n,miniBatchSize) ]
            for batch in miniBatches:
                self.update_mini_batch(batch,learnRate)
            print("Epoch {0} complete.".format(j))

    #-----------------
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
                self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(z, self):
        return self.sigmoid(z) * (1 - self.sigmoid(z))