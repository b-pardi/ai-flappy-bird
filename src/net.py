import numpy as np
import random

'''
                 +----------------+
                 |     Output     |                             output E [0,1]
                 |      (o)       |
                 +----------------+
                         ^
                         |
                         |
                 +----------------+
                 |   Sum & Act    |
                 |       (Σa)     |<----------------+
                 +----------------+                 |
                 ^       ^        ^                 |
                /        |         \                |
               /         |          \               |
              /          |           \              |
             /           |            \             |
       +-------+    +-------+     +-------+     +-------+
       |  w0   |    |  w1   |     |  w2   |     |  w3   |       w E [-1:1]
       +-------+    +-------+     +-------+     +-------+
           ^            ^             ^             ^
           |            |             |             |
      +--------+----+-------+------+------+------+--------+
      |   i0     |     i1      |      i2      |     i3=1  |
      | (Input)  |   (Input)   |   (Input)    |    (Bias) |
      +--------+--------+--------+--------+--+------------+
'''



class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight

class Neuron:
    n_layrs = 2 # input and output layers
    def __init__(self, id):
        self.id = id
        self.layer = 0 # layer of perceptron node is in
        self.input = 0
        self.output = 0
        self.connections = []

    def activate(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        if self.layer != 0:
            self.output = sigmoid(self.input)
        
        # add product of weight and neuron's output value of each to neuron to neuron its connected to
        for wire in self.connections:
            weighted_output = wire.weight * self.output
            wire.to_neuron.input += weighted_output

class Net:
    n_layers = 2
    def __init__(self, n_inputs):
        self.connections = []
        self.neurons = []
        self.n_inputs = n_inputs # what the bird sees + bias
        self.net = []

        # create input nodes
        for i in range(self.n_inputs): # 4 inputs
            self.neurons.append(Neuron(i))
            self.neurons[i].layer = 0
        
        # bias neuron
        self.neurons.append(Neuron(3))
        self.neurons[self.n_inputs].layer = 0

        # output neuron
        self.neurons.append(Neuron(4))
        self.neurons[self.n_inputs+1].layer=1

        # connections
        output_neuron = self.neurons[-1]
        for i in range(0,len(self.neurons)-1):
            self.connections.append(Connection(self.neurons[i], output_neuron, random.uniform(-1,1)))

    def connect_neurons(self):
        for i in range(0, len(self.neurons)):
            self.neurons[i].connections = [] # clear previous connections

        # 
        for i in range(0, len(self.connections)):
            self.connections[i].from_neuron.connections.append(self.connections[i])

    def generate_net(self):
        self.connect_neurons()
        self.net = [] # clear out previous network

        # fill net with nodes ordered by their layer
        for i in range(0, self.n_layers):
            for j in range(0, len(self.neurons)):
                if self.neurons[j].layer == i:
                    self.net.append(self.neurons[j])

    def ff(self, vision):
        for i in range(0, self.n_inputs):
            self.neurons[i].output = vision[i]
        self.neurons[self.n_inputs + 1].output = 1 # set bias to 1

        for neuron in self.net:
            neuron.activate()

        output = self.neurons[-1].output

        # reset input values of neurons to 0
        for i in range(0, len(self.neurons)):
            self.neurons[i].input = 0

        return output