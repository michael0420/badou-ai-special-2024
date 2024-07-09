import numpy
import scipy.special

class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, lr=0.001):
        # init the neural network
        self.in_nodes = inputnodes
        self.hid_nodes = hiddennodes
        self.out_nodes = outputnodes

        self.lr = lr
        
        # init the weight
        self.weight_in_hid = (numpy.random.normal(0.0, pow(self.hid_nodes, -0.5), (self.hid_nodes, self.in_nodes)))
        self.weight_hid_out = (numpy.random.normal(0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.hid_nodes)))
        
        # init the activation function
        self.aFc = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        # train the neural network
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate the input to hidden layer
        hidden_inputs = numpy.dot(self.weight_in_hid, inputs)
        hidden_outputs = self.aFc(hidden_inputs)
        
        # calculate the hidden to output layer
        final_inputs = numpy.dot(self.weight_hid_out, hidden_outputs)
        final_outputs = self.aFc(final_inputs)
        
        # calculate the error
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.weight_hid_out.T, output_errors)
        
        # update the weight
        self.weight_hid_out += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.weight_in_hid += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        # query the neural network
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate the input to hidden layer
        hidden_inputs = numpy.dot(self.weight_in_hid, inputs)
        hidden_outputs = self.aFc(hidden_inputs)
        
        # calculate the hidden to output layer
        final_inputs = numpy.dot(self.weight_hid_out, hidden_outputs)
        final_outputs = self.aFc(final_inputs)
        
        return final_outputs

# set the parameter
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
epochs = 5
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist dataset
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
for e in range(epochs):
    for data in training_data_list:
        all_values = data.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

# load the test dataset
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for data in test_data_list:
    all_values = data.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if label == correct_label:
        scores.append(1)
    else:
        scores.append(0)

scores = numpy.array(scores)
print("performance = ", scores.sum() / scores.size)

    


