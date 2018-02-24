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

generated_ds_path = './generated_datasets/'
ds_path = './hwk3_datasets/'
ds_types = ['train', 'valid', 'test']
ds = ['yelp-', 'IMDB-']
classifier_average = 'micro'
num_features = 10000

def extract_features(set, n):
    file = ds_path + set + ds_types[0] + '.txt'
    reviews = []

    # find top features and write to file
    with open(file, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            sp = l.split('\t')
            reviews.append(re.compile('[^\w\s]').sub('', sp[0].strip()).lower())
    words = Counter([word for line in reviews for word in line.split()]).most_common(num_features)
    dictionary = {}
    writer = open(generated_ds_path + set.split('-')[0] + '-vocab.txt', 'w')
    for i in range(num_features):
        word = words[i][0]
        dictionary[word] = i + 1
        text = ("{}\t{}\t{}\n".format(word, i + 1, words[i][1]))
        writer.write(text)

    # create feature vectors and write to file
    for type in ds_types:
        file = ds_path + set + type + '.txt'
        translator = str.maketrans(" ", " ", string.punctuation)
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        examples = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator).split("\n")[:-1]
        output = [i[-1] for i in examples]
        writer = open(generated_ds_path + set.split('-')[0] + '-' + type.split('.')[0] + '.txt', 'w')
        for i in range(len(examples)):
            text = ''
            for word in examples[i].split(' ')[:-1]:
                if word in dictionary.keys():
                    text = '{} {}'.format(text, dictionary[word])
            if len(text) == 0: text = ' '
            text = "{}\t{}\n".format(text, output[i])
            writer.write(text[1:])

    return dictionary

def to_bow(features, set):
    binary = {}
    freq = {}
    for type in ds_types:
        # read text from file
        file = ds_path + set + type + '.txt'
        translator = str.maketrans(" ", " ", string.punctuation)
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        text = list(filter(None, text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator).split('\n')))
        output = np.array([int(line[-1]) for line in text])
        input = [line[:-1] for line in text]
        # use vectorizer object create binary and frequency bag of words
        vectorizer = CountVectorizer(vocabulary=features.keys())
        vectors = np.asarray(vectorizer.fit_transform(input).todense())
        freq[type] = [sparse.csr_matrix(normalize(vectors)), output]
        vectors[vectors > 1] = 1
        binary[type] = [sparse.csr_matrix(vectors), output]
    return binary, freq

def random_classifier(set):
    classifier = DummyClassifier(strategy="uniform")
    classifier.fit(set['train'][0], set['train'][1])
    prediction_train = f1_score(set['train'][1], classifier.predict(set['train'][0]), average=classifier_average)
    prediction_valid = f1_score(set['valid'][1], classifier.predict(set['valid'][0]), average=classifier_average)
    prediction_test = f1_score(set['test'][1], classifier.predict(set['test'][0]), average=classifier_average)
    return prediction_train, prediction_valid, prediction_test

def majority_classifier(set):
    classifier = DummyClassifier(strategy="most_frequent")
    classifier.fit(set['train'][0], set['train'][1])
    prediction_train = f1_score(set['train'][1], classifier.predict(set['train'][0]), average=classifier_average)
    prediction_valid = f1_score(set['valid'][1], classifier.predict(set['valid'][0]), average=classifier_average)
    prediction_test = f1_score(set['test'][1], classifier.predict(set['test'][0]), average=classifier_average)
    return prediction_train, prediction_valid, prediction_test


def train_model(set, classifier, params):

    if params != None:
        split = PredefinedSplit(test_fold=[-1 for i in range(set['train'][0].shape[0])] + [0 for i in range(set['valid'][0].shape[0])])
        classifier = GridSearchCV(classifier, params, cv=split, refit=True)
        merged_input = sparse.vstack([set['train'][0], set['valid'][0]])
        merged_output = np.concatenate((set['train'][1], set['valid'][1]))
        classifier.fit(merged_input, merged_output)
    else:
        classifier.fit(set['train'][0], set['train'][1])

    prediction_train = f1_score(set['train'][1], classifier.predict(set['train'][0]), average=classifier_average)
    prediction_valid = f1_score(set['valid'][1], classifier.predict(set['valid'][0]), average=classifier_average)
    prediction_test = f1_score(set['test'][1], classifier.predict(set['test'][0]), average=classifier_average)
    best_param = None if params == None else classifier.best_params_

    return prediction_train, prediction_valid, prediction_test, best_param


''' Part 1 - Cleaning and Vectorizing Data \n '''
print('Part 1 - Cleaning and Vectorizing Data \n')

# Clean and vectorize yelp data
features = extract_features(ds[0], num_features)
yelp_binary, yelp_freq = to_bow(features, ds[0])

# Clean and vectorize IMDB data
features = extract_features(ds[1], num_features)
imdb_binary, imdb_freq = to_bow(features, ds[1])

print('Dataset cleaned and vectorized. Results can be found in generated folder \n')

''' Part 2 - Yelp Classification Using Binary Bag Of Words Representation \n '''
print('Part 2 - Yelp Classification Using Binary Bag Of Words Representation \n')

# Random Classifier
f1_prediction = random_classifier(yelp_binary)
print('Random Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Majority Classifier
f1_prediction = majority_classifier(yelp_binary)
print('Majority Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(yelp_binary, BernoulliNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
param = [{'max_depth': [i for i in range(14, 18)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [1000 * i for i in range(1,5)]}]
f1_prediction = train_model(yelp_binary, DecisionTreeClassifier(), param)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
param = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(yelp_binary, LinearSVC(), param)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}'.format(f1_prediction[3]))

''' Part 3 - Yelp Classification Using Frequency Bag Of Words Representation \n '''
print('Part 2 - Yelp Classification Using Frequency Bag Of Words Representation \n')

# Random Classifier
f1_prediction = random_classifier(yelp_freq)
print('Random Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Majority Classifier
f1_prediction = majority_classifier(yelp_freq)
print('Majority Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(yelp_freq, BernoulliNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
param = [{'max_depth': [i for i in range(14, 18)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [1000 * i for i in range(1,5)]}]
f1_prediction = train_model(yelp_freq, DecisionTreeClassifier(), param)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
param = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(yelp_freq, LinearSVC(), param)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}'.format(f1_prediction[3]))

''' Part 4 (I) - IMDB Classification Using Binary Bag Of Words Representation \n '''
print('Part 4 (I) - IMDB Classification Using Binary Bag Of Words Representation \n')

# Random Classifier
f1_prediction = random_classifier(imdb_binary)
print('Random Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(imdb_binary, GaussianNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
param = [{'max_depth': [i for i in range(14, 18)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [1000 * i for i in range(1,5)]}]
f1_prediction = train_model(imdb_binary, DecisionTreeClassifier(), param)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
param = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(imdb_binary, LinearSVC(), param)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}'.format(f1_prediction[3]))

''' Part 4 (II) - IMDB Classification Using Frequency Bag Of Words Representation \n '''
print('Part 4 (II) - IMDB Classification Using Frequency Bag Of Words Representation \n')

# Random Classifier
f1_prediction = random_classifier(imdb_binary)
print('Random Classifier \n(train, valid, test): {} \n'.format(f1_prediction))

# Naive Bayes
params = [{'alpha': np.arange(0.4, 0.6, 0.8)}]
f1_prediction = train_model(imdb_binary, GaussianNB(), params)
print('Naive Bayes Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Decision Tree
param = [{'max_depth': [i for i in range(14, 18)], 'max_features': [1000 * i for i in range(4, 8)], 'max_leaf_nodes': [1000 * i for i in range(1,5)]}]
f1_prediction = train_model(imdb_binary, DecisionTreeClassifier(), param)
print('Decision Tree \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}\n'.format(f1_prediction[3]))

# Linear SVM
param = [{'max_iter': [100 * i for i in range(10)]}]
f1_prediction = train_model(imdb_binary, LinearSVC(), param)
print('Linear SVM Classifier \n(train, valid, test) = {}'.format(f1_prediction[:3]))
print('best parameters = {}'.format(f1_prediction[3]))
