import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TRAIN_DATA = 'data/Dataset_1_train.csv'
VALID_DATA = 'data/Dataset_1_valid.csv'
TEST_DATA = 'data/Dataset_1_test.csv'
PLOT_PRECISION = 60
POLY_DEGREE = 20
NUM_LAMBDA_VALUE = 100

''' Computes the polynomial matrix of a set of data '''
def findPolyMatrix(data_frame, degree):
    data_x = data_frame[0]
    data_y = data_frame[1]
    # create polynomial matrix
    data_frame.drop([1, 2], axis=1, inplace=True)
    for i in range(degree + 1):
        data_frame[i] = np.power(data_x, i)
    # return polynomial and output matrices
    return data_frame, data_y

''' Computes the matrix of weighted coefficients
     W = inv[X^(t)X + LI][X^(t)Y]
     where X = Polynomial Matrix, L=Lambda, I=Identity Matrix, Y=Output Matrix '''
def findWeightedMatrix(poly_matrix, output_matrix, lambd, degree):
    lambd_matrix = np.identity(degree+1) * lambd # LI
    XtX_inv = np.linalg.pinv(np.matmul(poly_matrix.T, poly_matrix) + lambd_matrix) # inv[X^(t)X + LI]
    XtY = np.matmul(poly_matrix.T, output_matrix) # [X^(t)Y]
    w = np.matmul(XtX_inv, XtY)
    return w

''' Computes the mean squared error of a weighted matrix on a data set '''
def findMSE(poly_matrix, weighted_matrix, output_matrix):
    sq_error = np.power(np.subtract(np.matmul(poly_matrix, weighted_matrix), output_matrix), 2)
    return np.sum(sq_error) / sq_error.size

''' Visualizes the data fit on a graph '''
def visualizeDataFit(poly_matrix, weighted_matrix, output_matrix, title, degree):
    x_axis = pd.DataFrame(np.ones(PLOT_PRECISION))
    x_axis[1] = np.arange(-1, 1, 2 / PLOT_PRECISION)
    for i in range(2, degree+1):
        x_axis[i] = pow(x_axis[1], i)
    decision_boundary = np.matmul(x_axis, weighted_matrix)
    plt.scatter(poly_matrix[1], output_matrix)
    plt.plot(x_axis[1], decision_boundary, 'r--')
    plt.title(title)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.tight_layout()
    plt.show()

# compute training/validation/testing matrices
train_poly_matrix, train_output_matrix = findPolyMatrix(pd.read_csv(TRAIN_DATA, header=None), POLY_DEGREE)
valid_poly_matrix, valid_output_matrix = findPolyMatrix(pd.read_csv(VALID_DATA, header=None), POLY_DEGREE)
test_poly_matrix, test_output_matrix = findPolyMatrix(pd.read_csv(TEST_DATA, header=None), POLY_DEGREE)

# 20-degree polynomial without regularization
w_matrix_no_reg = findWeightedMatrix(train_poly_matrix, train_output_matrix, 0, POLY_DEGREE)
mse_no_reg = findMSE(valid_poly_matrix, w_matrix_no_reg, valid_output_matrix)
visualizeDataFit(train_poly_matrix, w_matrix_no_reg, train_output_matrix, 'Training Data', POLY_DEGREE)
visualizeDataFit(valid_poly_matrix, w_matrix_no_reg, valid_output_matrix, 'Validation Data', POLY_DEGREE)
print('MSE without regularization: ')
print(mse_no_reg)

# 20-degree polynomial with regularization
best_lambda = 0.0
best_mse = 10000
interval = 1 / NUM_LAMBDA_VALUE
lamba = 0.0
for i in range(NUM_LAMBDA_VALUE):
    w_matrix_reg = findWeightedMatrix(train_poly_matrix, train_output_matrix, lamba, POLY_DEGREE)
    mse_reg = findMSE(valid_poly_matrix, w_matrix_reg, valid_output_matrix)
    if best_mse > mse_reg:
        best_lambda = lamba
        best_mse = mse_reg
    lamba += interval
print('Best Lambda:')
print(best_lambda)
print('Best MSE:')
print(best_mse)



