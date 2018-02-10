import numpy as np
import pandas as pd
import math

POS_MEAN_1 = 'data/DS2_c1_m1.txt'
POS_MEAN_2 = 'data/DS2_c1_m2.txt'
POS_MEAN_3 = 'data/DS2_c1_m3.txt'
NEG_MEAN_1 = 'data/DS2_c2_m1.txt'
NEG_MEAN_2 = 'data/DS2_c2_m2.txt'
NEG_MEAN_3 = 'data/DS2_c2_m3.txt'
COV_1 = 'data/DS2_Cov1.txt'
COV_2 = 'data/DS2_Cov2.txt'
COV_3 = 'data/DS2_Cov3.txt'
TRAIN_DATA = 'generated_data/DS2_train.csv'
TEST_DATA = 'generated_data/DS2_test.csv'

''' Part 4 '''
print('Part 4 - Generating Dataset 2 Using 3 Different Gaussian Distributions \n')
# Prepare Data
mean0_1 = pd.read_csv(NEG_MEAN_1, header=None)
mean1_1 = pd.read_csv(POS_MEAN_1, header=None)
mean0_2 = pd.read_csv(NEG_MEAN_2, header=None)
mean1_2 = pd.read_csv(POS_MEAN_2, header=None)
mean0_3 = pd.read_csv(NEG_MEAN_3, header=None)
mean1_3 = pd.read_csv(POS_MEAN_3, header=None)
cov_1 = pd.read_csv(COV_1, header=None)
cov_2 = pd.read_csv(COV_2, header=None)
cov_3 = pd.read_csv(COV_3, header=None)
mean0_1.drop([20], axis=1, inplace=True)
mean1_1.drop([20], axis=1, inplace=True)
cov_1.drop([20], axis=1, inplace=True)
mean0_2.drop([20], axis=1, inplace=True)
mean1_2.drop([20], axis=1, inplace=True)
cov_2.drop([20], axis=1, inplace=True)
mean0_3.drop([20], axis=1, inplace=True)
mean1_3.drop([20], axis=1, inplace=True)
cov_3.drop([20], axis=1, inplace=True)

# Generate samples using gaussian distribution
data0_1 = pd.DataFrame(np.random.multivariate_normal(mean0_1.as_matrix()[0], cov_1.values, 200))
data1_1 = pd.DataFrame(np.random.multivariate_normal(mean1_1.as_matrix()[0], cov_1.values, 200))
data0_2 = pd.DataFrame(np.random.multivariate_normal(mean0_2.as_matrix()[0], cov_2.values, 840))
data1_2 = pd.DataFrame(np.random.multivariate_normal(mean1_2.as_matrix()[0], cov_2.values, 840))
data0_3 = pd.DataFrame(np.random.multivariate_normal(mean0_3.as_matrix()[0], cov_3.values, 960))
data1_3 = pd.DataFrame(np.random.multivariate_normal(mean1_3.as_matrix()[0], cov_3.values, 960))
data0_1[20] = 0
data1_1[20] = 1
data0_2[20] = 0
data1_2[20] = 1
data0_3[20] = 0
data1_3[20] = 1

# Split data into test and training
msk = np.random.rand(len(data0_1)) < 0.7
train_data0_1 = data0_1.loc[msk]
test_data0_1 = data0_1.loc[~msk]
msk = np.random.rand(len(data1_1)) < 0.7
train_data1_1 = data1_1.loc[msk]
test_data1_1 = data1_1.loc[~msk]
msk = np.random.rand(len(data0_2)) < 0.7
train_data0_2 = data0_2.loc[msk]
test_data0_2 = data0_2.loc[~msk]
msk = np.random.rand(len(data1_2)) < 0.7
train_data1_2 = data1_2.loc[msk]
test_data1_2 = data1_2.loc[~msk]
msk = np.random.rand(len(data0_3)) < 0.7
train_data0_3 = data0_3.loc[msk]
test_data0_3 = data0_3.loc[~msk]
msk = np.random.rand(len(data1_3)) < 0.7
train_data1_3 = data1_3.loc[msk]
test_data1_3 = data1_3.loc[~msk]

# Put both classes together in single data set
train_data = pd.concat([train_data0_1, train_data1_1, train_data0_2, train_data1_2, train_data0_3, train_data1_3], ignore_index=True)
test_data = pd.concat([test_data0_1, test_data1_1, test_data0_2, test_data1_2, test_data0_3, test_data1_3], ignore_index=True)
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

''' Part 5.1 '''
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

''' Part 5.2 '''
print('Part 3 - k-NN Classifier \n')

# Variables to keep track of result from each step
confusion_steps = []
accuracy_steps = []

# Perform knn classifier
for i in range(1, 25):
    confusion = [[0, 0], [0, 0]]
    for j in range(len(test_data)):
        distances = np.array(np.power(abs(train_data.sub(np.array(np.array(test_data.loc[[j], :])[0]))), 2).sum(axis=1))
        closest_neighbours = np.array([train_output[j] for j in np.argpartition(distances, i)[:i]])
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