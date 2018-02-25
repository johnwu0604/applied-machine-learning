import re, string
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

GENERATED_DATA = './generated_datasets/'
DS_PATH = './hwk3_datasets/'
DS_TYPES = ['train', 'valid', 'test']
DS = ['yelp-', 'IMDB-']
AVERAGE = 'micro'
NUM_FEATURES = 10000

# Extract the top features from the dataset and write them to a file
def extract_features(dataset):
    file = DS_PATH + dataset + DS_TYPES[0] + '.txt'
    reviews = []

    # find top features and write to file
    with open(file, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            sp = l.split('\t')
            reviews.append(re.compile('[^\w\s]').sub('', sp[0].strip()).lower())
    words = Counter([word for line in reviews for word in line.split()]).most_common(NUM_FEATURES)
    dictionary = {}
    writer = open(GENERATED_DATA + dataset.split('-')[0] + '-vocab.txt', 'w')
    for i in range(NUM_FEATURES):
        word = words[i][0]
        dictionary[word] = i + 1
        text = ("{}\t{}\t{}\n".format(word, i + 1, words[i][1]))
        writer.write(text)

    # create feature vectors and write to file
    for type in DS_TYPES:
        file = DS_PATH + dataset + type + '.txt'
        translator = str.maketrans(" ", " ", string.punctuation)
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        examples = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator).split("\n")[:-1]
        output = [i[-1] for i in examples]
        writer = open(GENERATED_DATA + dataset.split('-')[0] + '-' + type.split('.')[0] + '.txt', 'w')
        for i in range(len(examples)):
            text = ''
            for word in examples[i].split(' ')[:-1]:
                if word in dictionary.keys():
                    text = '{} {}'.format(text, dictionary[word])
            if len(text) == 0: text = ' '
            text = "{}\t{}\n".format(text, output[i])
            writer.write(text[1:])

    return dictionary

# Coverts a dataset to bag of words representation
def to_bow(features, set):
    binary = {}
    freq = {}
    for type in DS_TYPES:
        # read text from file
        file = DS_PATH + set + type + '.txt'
        translator = str.maketrans(" ", " ", string.punctuation)
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        text = list(filter(None, text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator).split('\n')))
        input = [line[:-1] for line in text]
        output = np.array([int(line[-1]) for line in text])
        # use vectorizer object create binary and frequency bag of words
        vectorizer = CountVectorizer(vocabulary=features.keys())
        vectors = np.asarray(vectorizer.fit_transform(input).todense())
        freq[type] = [sparse.csr_matrix(normalize(vectors)), output]
        vectors[vectors > 1] = 1
        binary[type] = [sparse.csr_matrix(vectors), output]
    return binary, freq

# Trains a model using the specified classification
def train_model(dataset, classifier, params):

    if params != None:
        split = PredefinedSplit(test_fold=[-1 for i in range(dataset['train'][0].shape[0])] + [0 for i in range(dataset['valid'][0].shape[0])])
        classifier = GridSearchCV(classifier, params, cv=split, refit=True)
        merged_input = sparse.vstack([dataset['train'][0], dataset['valid'][0]])
        merged_output = np.concatenate((dataset['train'][1], dataset['valid'][1]))
        classifier.fit(merged_input, merged_output)
    else:
        classifier.fit(dataset['train'][0], dataset['train'][1])

    prediction_train = f1_score(dataset['train'][1], classifier.predict(dataset['train'][0]), average=AVERAGE)
    prediction_valid = f1_score(dataset['valid'][1], classifier.predict(dataset['valid'][0]), average=AVERAGE)
    prediction_test = f1_score(dataset['test'][1], classifier.predict(dataset['test'][0]), average=AVERAGE)
    best_param = None if params == None else classifier.best_params_

    return prediction_train, prediction_valid, prediction_test, best_param


''' Part 1 - Cleaning and Vectorizing Data \n '''
print('Part 1 - Cleaning and Vectorizing Data \n')

# Clean and vectorize yelp data
features = extract_features(DS[0])
yelp_binary, yelp_freq = to_bow(features, DS[0])

# Clean and vectorize IMDB data
features = extract_features(DS[1])
imdb_binary, imdb_freq = to_bow(features, DS[1])

print('Dataset cleaned and vectorized. Results can be found in generated folder. \n')

''' Part 2 - Yelp Classification Using Binary Bag Of Words Representation \n '''
print('Part 2 - Yelp Classification Using Binary Bag Of Words Representation \n')

# Random Classifier
f1_prediction = train_model(yelp_binary, DummyClassifier(strategy="uniform"), None)
print('Random Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Majority Classifier
f1_prediction = train_model(yelp_binary, DummyClassifier(strategy="most_frequent"), None)
print('Majority Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(yelp_binary, BernoulliNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
params = [{'max_depth': [i for i in range(10, 25)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [100 * i for i in range(1,10)]}]
f1_prediction = train_model(yelp_binary, DecisionTreeClassifier(), params)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
params = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(yelp_binary, LinearSVC(), params)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {} \n'.format(f1_prediction[3]))

''' Part 3 - Yelp Classification Using Frequency Bag Of Words Representation \n '''
print('Part 3 - Yelp Classification Using Frequency Bag Of Words Representation \n')

# Random Classifier
f1_prediction = train_model(yelp_freq, DummyClassifier(strategy="uniform"), None)
print('Random Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Majority Classifier
f1_prediction = train_model(yelp_freq, DummyClassifier(strategy="most_frequent"), None)
print('Majority Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Decision Tree
params = [{'max_depth': [i for i in range(10, 25)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [100 * i for i in range(1,10)]}]
f1_prediction = train_model(yelp_freq, DecisionTreeClassifier(), params)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
params = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(yelp_freq, LinearSVC(), params)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {} \n'.format(f1_prediction[3]))

# Naive Bayes
yelp_freq['train'][0] = yelp_freq['train'][0].todense()
yelp_freq['valid'][0] = yelp_freq['valid'][0].todense()
yelp_freq['test'][0] = yelp_freq['test'][0].todense()
f1_prediction = train_model(yelp_freq, GaussianNB(), None)
print('Naive Bayes\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

''' Part 4 (I) - IMDB Classification Using Binary Bag Of Words Representation \n '''
print('Part 4 (I) - IMDB Classification Using Binary Bag Of Words Representation \n')

# Random Classifier
f1_prediction = train_model(imdb_binary, DummyClassifier(strategy="uniform"), None)
print('Random Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(imdb_binary, BernoulliNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
params = [{'max_depth': [i for i in range(10, 25)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [100 * i for i in range(1,10)]}]
f1_prediction = train_model(imdb_binary, DecisionTreeClassifier(), params)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
params = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(imdb_binary, LinearSVC(), params)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {} \n'.format(f1_prediction[3]))

''' Part 4 (II) - IMDB Classification Using Frequency Bag Of Words Representation \n '''
print('Part 4 (II) - IMDB Classification Using Frequency Bag Of Words Representation \n')

# Random Classifier
f1_prediction = train_model(imdb_freq, DummyClassifier(strategy="uniform"), None)
print('Random Classifier\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))

# Decision Tree
params = [{'max_depth': [i for i in range(10, 25)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [100 * i for i in range(1,10)]}]
f1_prediction = train_model(imdb_freq, DecisionTreeClassifier(), params)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
params = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(imdb_freq, LinearSVC(), params)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {} \n'.format(f1_prediction[3]))

# Naive Bayes
imdb_freq['train'][0] = imdb_freq['train'][0].todense()
imdb_freq['valid'][0] = imdb_freq['valid'][0].todense()
imdb_freq['test'][0] = imdb_freq['test'][0].todense()
f1_prediction = train_model(imdb_freq, GaussianNB(), None)
print('Naive Bayes\n(train, valid, test) = {} \n'.format(f1_prediction[:3]))
