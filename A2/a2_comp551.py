import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NEG_MEAN_1 = 'data/DS1_m_0.txt'
POS_MEAN_1 = 'data/DS1_m_1.txt'
COV_1 = 'data/DS1_Cov.txt'
TRAIN_DATA = 'generated_data/DS1_train.csv'
TEST_DATA = 'generated_data/DS1_test.csv'

''' Part 1 '''
print('Part 1 - Generating Data Using Dataset 1 Mean and Covariances ... \n')
# Prepare Data
neg_mean = pd.read_csv(NEG_MEAN_1, header=None)
pos_mean = pd.read_csv(POS_MEAN_1, header=None)
cov = pd.read_csv(COV_1, header=None)
neg_mean.drop([20], axis=1, inplace=True)
pos_mean.drop([20], axis=1, inplace=True)
cov.drop([20], axis=1, inplace=True)

# Generate samples using gaussian distribution
neg_values = np.random.multivariate_normal(neg_mean.as_matrix()[0], cov.values, 2000)
pos_values = np.random.multivariate_normal(pos_mean.as_matrix()[0], cov.values, 2000)
neg_values = pd.DataFrame(neg_values)
pos_values = pd.DataFrame(pos_values)
neg_values[20] = 0
pos_values[20] = 1

# Split data into test and training
msk = np.random.rand(len(neg_values)) < 0.7
neg_values_train = neg_values[msk]
neg_values_test = neg_values[~msk]
msk = np.random.rand(len(pos_values)) < 0.7
pos_values_train = pos_values[msk]
pos_values_test = pos_values[~msk]

# Put both classes together in single data set
train_data = pd.concat([neg_values_train, pos_values_train], ignore_index=True)
test_data = pd.concat([neg_values_test, pos_values_test], ignore_index=True)
train_data.to_csv(TRAIN_DATA)
test_data.to_csv(TEST_DATA)

print('Training Data Generated. See: {}'.format(TRAIN_DATA))
print('Testing Data Generated. See: {}'.format(TEST_DATA))

''' Part 2 '''