import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

TRAIN_DATA = 'data/Dataset_2_train.csv'
VALID_DATA = 'data/Dataset_2_valid.csv'
TEST_DATA = 'data/Dataset_2_test.csv'

ERROR_THRESHOLD = 0.1

''' Parses the data file into a data frame object '''
def parseData(file):
    data_frame = pd.read_csv(file, header=None)
    data_frame.drop([2], axis=1, inplace=True)
    return data_frame

''' Computes the output of a given input using indicated parameters '''
def computeOutput(w0, w1, x):
    return w0 + w1 * x

''' Performs one full epoche of the data '''
def performEpoche(train_data, w0, w1, step_size):
    for i in range(train_data.shape[0]):
        predicted_output = computeOutput(w0, w1, train_data[0][i])
        actual_output = train_data[1][i]
        loss = predicted_output - actual_output
        w0 = w0 - step_size * loss
        w1 = w1 - step_size * loss * train_data[0][i]
    return w0, w1

''' Compute the mean squared error '''
def computeMSE(w0, w1, test_data):
    total_error = 0
    for i in range(test_data.shape[0]):
        predicted_output = computeOutput(w0, w1, test_data[0][i])
        actual_output = test_data[1][i]
        total_error += np.power(predicted_output - actual_output, 2)
    return total_error / test_data.shape[0]

''' Visualize the learning curve '''
def visualizeLearning(iteration_values, mse_values, title):
    plt.scatter(iteration_values, mse_values)
    plt.plot(iteration_values, mse_values)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.show()

''' Part 1 '''
train_data = parseData(TRAIN_DATA)
valid_data = parseData(VALID_DATA)
step_size = 1e-6
w0, w1 = random.randint(0, 10), random.randint(0, 10)
print('Initial w0: ', w0, ', Initial w1: ', w1)
iteration_values, mse_train_values, mse_valid_values = [], [], []
for i in range(1, 10000):
    w0, w1 = performEpoche(train_data, w0, w1, step_size)
    mse_train_values.append(computeMSE(w0, w1, train_data))
    mse_valid_values.append(computeMSE(w0, w1, valid_data))
    iteration_values.append(i)
visualizeLearning(iteration_values, mse_train_values, 'Learning Curve (Training Data)')
visualizeLearning(iteration_values, mse_valid_values, 'Learning Curve (Validation Data)')
print('Final w0: ', w0, ', Final w1: ', w1)
print('MSE (Training Data): ', mse_train_values[10000-1])
print('MSE (Validation Data): ', mse_valid_values[10000-1])

