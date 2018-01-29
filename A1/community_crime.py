import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = 'community-crime/'
STEP_SIZE = 1e-5

''' Prepare our data by cleaning and splitting it '''
def prepareData():
    # clean data
    df = pd.read_csv('{}raw_data.csv'.format(DATA_DIR), header=None)
    df.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
    df = df.replace('?', np.NaN).astype(np.float64)
    df.fillna(df.mean(), inplace=True)
    df.to_csv('{}cleaned_data.csv'.format(DATA_DIR), index=False, header=False)
    # split data
    for i in range(1, 6):
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        train.to_csv('{}CandC−train{}.csv'.format(DATA_DIR, i), index=False, header=False)
        test.to_csv('{}CandC−test{}.csv'.format(DATA_DIR, i), index=False, header=False)

''' Creates the initial matrices needed for the output computation '''
def createComputationMatrices(train_data, test_data):
    # create vector
    train_data_output = pd.DataFrame([train_data[train_data.shape[1] - 1]])
    test_data_output = pd.DataFrame([test_data[test_data.shape[1] - 1]])
    # modify input matrix (shift and add column of ones)
    for i in reversed(range(1, train_data.shape[1])):
        train_data[i] = train_data[i - 1]
    train_data.drop([0], axis=1, inplace=True)
    train_data.insert(0, 0, np.ones(train_data.shape[0]))
    for i in reversed(range(1, test_data.shape[1])):
        test_data[i] = test_data[i - 1]
    test_data.drop([0], axis=1, inplace=True)
    test_data.insert(0, 0, np.ones(test_data.shape[0]))
    # create w vector
    w = pd.DataFrame(np.random.randint(low=0, high=10, size=(1, train_data.shape[1])))
    return train_data, train_data_output, test_data, test_data_output, w

''' Find the weighted matrix '''
def findWeightedMatrix(input_values, output_values, w, lambd):
    lambd_matrix = np.identity(input_values.shape[1]) * lambd
    inv_matrix = np.linalg.pinv(np.matmul(input_values.T, input_values) + lambd_matrix)
    XtY = np.matmul(input_values.T, output_values.T)
    w = np.matmul(inv_matrix, XtY)
    return w

''' Calculates the mean squared area '''
def calculateMSE(w, test_data_input, test_data_output):
    predicted_output = np.matmul(w, test_data_input.T)
    squared_error = np.power(np.subtract(predicted_output, test_data_output), 2)
    return np.sum(np.sum(squared_error)) / squared_error.size

''' Performs the linear regression '''
def performRegression():
    mse_values, w_values = [], []
    for i in range(1, 6):
        train_data = pd.read_csv('{}CandC−train{}.csv'.format(DATA_DIR, i), header=None)
        test_data = pd.read_csv('{}CandC−test{}.csv'.format(DATA_DIR, i), header=None)
        train_data_input, train_data_output, test_data_input, test_data_output, w = createComputationMatrices(train_data, test_data)
        for j in range(100000):
            predicted_output = np.matmul(w, train_data_input.T)
            loss = np.matmul(np.subtract(predicted_output, train_data_output), train_data_input) / train_data_output.shape[0]
            w = w - STEP_SIZE * loss
        mse = calculateMSE(w, test_data_input, test_data_output)
        mse_values.append(mse)
        w_values.append(w.values.tolist()[0])
    return mse_values, w_values

''' Compute optimal parameters for lambda and its corresponding mse value'''
def performRidgeRegression():
    lambd_values = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    best_lambda = 0.0
    best_mse = 10000
    mse_values = []
    w_values = []
    for i in range(len(lambd_values)):
        total_mse = 0
        current_w_values = []
        for j in range(1, 6):
            train_data = pd.read_csv('{}CandC−train{}.csv'.format(DATA_DIR, j), header=None)
            test_data = pd.read_csv('{}CandC−test{}.csv'.format(DATA_DIR, j), header=None)
            train_data_input, train_data_output, test_data_input, test_data_output, w = createComputationMatrices(train_data, test_data)
            w = findWeightedMatrix(train_data_input, train_data_output, w, lambd_values[i])
            mse = calculateMSE(w.T, test_data_input, test_data_output)
            total_mse += mse
            current_w_values.append(w)
        average_mse = total_mse/5
        mse_values.append(average_mse)
        # calculate average parameters
        average_w = []
        for k in range(len(current_w_values[0])):
            average = 0
            for l in range(len(current_w_values)):
                average += current_w_values[l][k][0]
            average_w.append(average/5)
        w_values.append(average_w)
        if (average_mse < best_mse):
            best_mse = average_mse
            best_lambda = lambd_values[i]
    return mse_values, w_values, lambd_values, best_mse, best_lambda

# Part 1
print('Part 1 - Preparing Data \n')
print('Prepared data can be found in \{} folder \n'.format(DATA_DIR))
prepareData()

# Part 2
print('Part 2 - Linear Regression \n')
mse_value, w_values = performRegression()
print('5-fold cross-validation error: ', sum(mse_value)/5)
for i in range(5):
    print('Set {} parameters: '.format(i+1), w_values[i])

# Part 3
print('Part 3 - Ridge Regression\n')
mse_values, w_values, lambd_values, best_mse, best_lambd = performRidgeRegression()
for i in range(len(mse_values)):
    print('Lambda Value: ', lambd_values[i])
    print(', Average MSE: ', mse_values[i], '\n')
    print('Parameters: ', w_values[i], '\n')


