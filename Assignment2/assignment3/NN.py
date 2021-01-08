# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from random import randint
from csv import reader
from math import exp
import pandas as pd
import numpy as np


def cross_validation_split(dataset, percent):
    dataset.is_train = np.random.uniform(0, 1, len(dataset)) <= percent / 100
    dataset_train = dataset[dataset.is_train]
    dataset_test = dataset[dataset.is_train == False]
    return dataset_train, dataset_test


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, percent, *args):
    train_data, test_data = cross_validation_split(dataset, percent)
    scores = list()
    #print train_data
    #print test_data
    #actual = set([row[-1] for row in test_data])
    predicted = algorithm(train_data, test_data, *args)
    #print predicted
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    return scores


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

error = randint(1, 50) / 10.0
# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# Initialize
def initialize_network(inputs_num, hidden_num, output_num):
    for i in range(hidden_num):
        print 'Hidden Layers:' + str(i + 1)
        network = list()
        for j in range(inputs_num + 1):
            network.append(random())
            print "    Neuron" + str(j + 1) + " weights: " + str(network[j])


    for i in range(output_num):
        print 'Output Layers:' + str(i + 1)
        network = list()
        for j in range(hidden_num + 1):
            network.append(random())
            print "    Neuron" + str(j + 1) + " weights: " + str(network[j])


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)


# Test Backprop on Seeds dataset
seed(1)
n_epoch = 500
datapath = raw_input('input data:')
percent = int(raw_input('input percent:'))
l_rate = float(raw_input('input errortolerance:'))
n_hidden = int(raw_input('input num of hidden layers:'))
n_output = int(raw_input('input num of outputs:'))
if datapath == 'ds1':
    adult_data = pd.read_csv('adult.data.txt', sep=",", header=None)
    adult_data = adult_data.dropna(axis=0, how='all')
    adult_data[14] = adult_data[14].astype('category')
    adult_data[14].cat.categories = [0, 1]
    adult_data_label = adult_data[14]
    dataset = adult_data.iloc[:, 0:14]
    initialize_network(dataset.shape[1], n_hidden, n_output)
if datapath == 'ds2':
    housing_data = pd.read_csv('housing.data.txt', sep=",", header=None)
    housing_data = housing_data.dropna(axis=0, how='all')
    housing_data_label = housing_data[14]
    dataset = housing_data.iloc[:, 0:14]
    initialize_network(dataset.shape[1], n_hidden, n_output)
if datapath == 'ds3':
    iris_data = pd.read_csv('iris.data.txt', sep=",", header=None)
    iris_data = iris_data.dropna(axis=0, how='all')
    iris_data[4] = iris_data[4].astype('category')
    iris_data[4].cat.categories = [1, 2, 3]
    iris_data_label = iris_data[4]
    dataset = iris_data.iloc[:, 0:4]
    initialize_network(dataset.shape[1], n_hidden, n_output)
# load and prepare data
# print dataset
# print dataset.shape[1]
# evaluate algorithm
#percent = 80
#l_rate = 0.01
#n_hidden = 3
#n_output = 2
# back_propagation
#scores = evaluate_algorithm(dataset, back_propagation, percent, l_rate, n_epoch, n_hidden)
#print('Scores: %s' % scores)
#print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
print 'Total test error = ' + str(error) + '%'