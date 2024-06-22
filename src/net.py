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

    def mutate_weight(self):
        # 10% chance for large change
        if random.uniform(0,1) > 0.9: # 80% chance mutation
            self.weight = random.uniform(-1,1)
        else: # 90% chance for small change
            self.weight += random.gauss(0,1) / 10
            if self.weight > 1:
                self.weight = 1
            else:
                self.weight -1

    def clone(self, from_neuron, to_neuron):
        clone = Connection(from_neuron, to_neuron, self.weight) 
        return clone

class Neuron:
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

    def clone(self):
        clone = Neuron(self.id)
        clone.id = self.id # ???
        clone.layer = self.layer
        return clone

class Net:
    def __init__(self, n_inputs, is_clone=False):
        self.connections = []
        self.neurons = []
        self.n_layers = 2
        self.n_inputs = n_inputs # what the bird sees + bias
        self.net = []

        if not is_clone: # if cloning a bird
            # create input nodes
            for i in range(self.n_inputs): # 4 inputs
                self.neurons.append(Neuron(i))
                self.neurons[i].layer = 0
            
            # bias neuron
            self.neurons.append(Neuron(3))
            self.neurons[3].layer = 0

            # output neuron
            self.neurons.append(Neuron(4))
            self.neurons[4].layer=1

            # connections
            output_neuron = self.neurons[-1]
            for i in range(0,len(self.neurons)-1):
                self.connections.append(Connection(self.neurons[i], output_neuron, random.uniform(-1,1)))


    def connect_neurons(self):
        for i in range(0, len(self.neurons)):
            self.neurons[i].connections = [] # clear previous connections

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
        for i in range(0, self.n_inputs-1):
            self.neurons[i].output = vision[i]
        self.neurons[self.n_inputs].output = 1 # set bias to 1

        for neuron in self.net:
            neuron.activate()

        output = self.neurons[-1].output

        # reset input values of neurons to 0
        for i in range(0, len(self.neurons)):
            self.neurons[i].input = 0

        return output
    
    def clone(self):
        clone = Net(self.n_inputs, True)
        for neuron in self.neurons:
            clone.neurons.append(neuron.clone())
        
        for connect in self.connections:
            cloned_connection = connect.clone(clone.get_neuron(connect.from_neuron.id),
                                              clone.get_neuron(connect.to_neuron.id))
            clone.connections.append(cloned_connection)
        clone.n_layers = self.n_layers
        clone.connect_neurons()
        return clone
    
    def get_neuron(self, id):
        for neuron in self.neurons:
            if neuron.id == id:
                return neuron
            
    def mutate(self):
        if random.uniform(0,1) > 0.2: # 80% chance mutation
            for connection in self.connections:
                connection.mutate_weight()