import numpy as np
import scipy.special as spl


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        self.activation_function = lambda x:spl.expit(x)


    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_inputs
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        self.who += self.lr*np.dot((output_errors*final_outputs*(1-final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot((hidden_errors*hidden_outputs*(1-hidden_outputs)), np.transpose(inputs))


    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
    
if __name__ == "__main__":
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    learning_rate = 0.1
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    with open("第十周作业\dataset\mnist_train.csv") as f:
        training_data_list = f.readlines()
    epochs = 20
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    
    with open("第十周作业\dataset\mnist_test.csv") as f:
        test_data_list = f.readlines()
    
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("改图片对应的数字为:", correct_number)
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        print("网络认为图片的数字是:", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    scores_array = np.asarray(scores)
    print("perfermance = ",scores_array.sum() / scores_array.size)


