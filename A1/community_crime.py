import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = 'community-crime/'
STEP_SIZE = 1e-5

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

def createComputationMatrices(train_data, test_data):
    # create vector
    # train_data_output = train_data[train_data.shape[1] - 1]
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

def calculateMSE(w, test_data_input, test_data_output):
    predicted_output = np.matmul(w, test_data_input.T)
    squared_error = np.power(np.subtract(predicted_output, test_data_output), 2)
    return np.sum(np.sum(squared_error)) / squared_error.size

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


def main():
    prepareData()
    mse_value, w_values = performRegression()
    print(mse_value)
    print(w_values)

main()