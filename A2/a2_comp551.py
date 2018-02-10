import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

NEG_MEAN_1 = 'data/DS1_m_0.txt'
POS_MEAN_1 = 'data/DS1_m_1.txt'
COV_1 = 'data/DS1_Cov.txt'
TRAIN_DATA = 'generated_data/DS1_train.csv'
TEST_DATA = 'generated_data/DS1_test.csv'

''' Part 1 '''
print('Part 1 - Generating Data Using Dataset 1 Mean and Covariances ... \n')
# Prepare Data
mean0 = pd.read_csv(NEG_MEAN_1, header=None)
mean1 = pd.read_csv(POS_MEAN_1, header=None)
cov = pd.read_csv(COV_1, header=None)
mean0.drop([20], axis=1, inplace=True)
mean1.drop([20], axis=1, inplace=True)
cov.drop([20], axis=1, inplace=True)

# Generate samples using gaussian distribution
data0 = pd.DataFrame(np.random.multivariate_normal(mean0.as_matrix()[0], cov.values, 2000))
data1 = pd.DataFrame(np.random.multivariate_normal(mean1.as_matrix()[0], cov.values, 2000))
data0[20] = 0
data1[20] = 1

# Split data into test and training
msk = np.random.rand(len(data0)) < 0.7
train_data0 = data0.loc[msk]
test_data0 = data0.loc[~msk]
msk = np.random.rand(len(data1)) < 0.7
train_data1 = data1.loc[msk]
test_data1 = data1.loc[~msk]

# Put both classes together in single data set
train_data = pd.concat([train_data0, train_data1], ignore_index=True)
test_data = pd.concat([test_data0, test_data1], ignore_index=True)
train_data.to_csv(TRAIN_DATA)
test_data.to_csv(TEST_DATA)

print('Training Data Generated. See: {}'.format(TRAIN_DATA))
print('Testing Data Generated. See: {} \n'.format(TEST_DATA))

# Create variables for future usage
train_data0 = pd.DataFrame(train_data[train_data[20] == 0])
train_data1 = pd.DataFrame(train_data[train_data[20] == 1])
test_output = test_data[20]
train_output = train_data[20]

# Drop the outputs
train_data0.drop([20], axis=1, inplace=True)
train_data1.drop([20], axis=1, inplace=True)
test_data.drop([20], axis=1, inplace=True)
train_data.drop([20], axis=1, inplace=True)

''' Part 2 '''
print('Part 2 - LDA Model Using Maximum Likelihood Approach \n')

# Find max probability
prob0 = float(len(train_data0)) / float(len(train_data0) + len(train_data1))
prob1 = 1.0 - prob0

# Find mean
mean0 = np.array(train_data0.mean())
mean1 = np.array(train_data1.mean())

# Find covariance matrix
diff0 = np.array(train_data0 - mean0)
diff1 = np.array(train_data1 - mean1)
cov = (np.matmul(diff0.T, diff0) + np.matmul(diff1.T, diff1)) / float(len(train_data0) + len(train_data1))

# Compute coefficients
w0 = math.log(prob0) - math.log(prob1) - 0.5 * (np.matmul(np.matmul(mean0.T, np.linalg.pinv(cov)), mean0) - np.matmul(np.matmul(mean1.T, np.linalg.pinv(cov)), mean1))
w1 = np.matmul(np.linalg.pinv(cov), mean0 - mean1)
print("w0: ", w0, '\n')
print("w1: " + str([i for i in w1]) + "\n")

# Compute output prediction
pred_output = np.matmul(test_data, w1) + w0

# Set prediction to 0 or 1 based on decision boundary
pred_output[pred_output > 0] = 0
pred_output[pred_output < 0] = 1

# Compute confusion matrix
confusion = [[0, 0], [0, 0]] # [[tp, fp],[fn, tn]]
for i in range(len(test_output)):
    true_value = test_output[i]
    pred_value = pred_output[i]
    if pred_value == 1:
        if pred_value == true_value:
            confusion[0][0] += 1
        else:
            confusion[0][1] += 1
    if pred_value == 0:
        if pred_value == true_value:
            confusion[1][1] += 1
        else:
            confusion[1][0] += 1
tp = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]
tn = confusion[1][1]

# Compute result
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f = 2 * precision * recall / (precision + recall)

print('F Measure: ', f)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall, '\n')

''' Part 3 '''
print('Part 3 - k-NN Classifier \n')

# Variables to keep track of result from each step
confusion_steps = []
accuracy_steps = []

# Perform knn classifier
for i in range(1, 25):
    confusion = [[0, 0], [0, 0]]
    for j in range(len(test_data)):
        distances = np.array(np.power(abs(train_data.sub(np.array(np.array(train_data.loc[[j], :])[0]))), 2).sum(axis=1))
        closest_neighbours_indices = np.argpartition(distances, i)[:i]
        closest_neighbours = np.array([train_output[k] for k in closest_neighbours_indices])
        pred_value = 1 if closest_neighbours.mean() > 0.5 else 0
        true_value = test_output[j]
        if pred_value == 1:
            if pred_value == true_value:
                confusion[0][0] += 1
            else:
                confusion[0][1] += 1
        if pred_value == 0:
            if pred_value == true_value:
                confusion[1][1] += 1
            else:
                confusion[1][0] += 1
    tp = confusion[0][0]
    fp = confusion[0][1]
    fn = confusion[1][0]
    tn = confusion[1][1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    confusion_steps.append(confusion)
    accuracy_steps.append(accuracy)

# Calculate optimal k
k_index = accuracy_steps.index(max(accuracy_steps))
print("Best K Value: ", k_index + 1, '\n')

# Calculate metrics
confusion = confusion_steps[k_index]
tp = confusion[0][0]
fp = confusion[0][1]
fn = confusion[1][0]
tn = confusion[1][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f = 2 * precision * recall / (precision + recall)
print('F Measure: ', f)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall, '\n')