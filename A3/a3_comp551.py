import re, string
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import PredefinedSplit
from sklearn.metrics import f1_score

YELP_TRAIN_DATA = 'hwk3_datasets/yelp-train.txt'
YELP_VALID_DATA = 'hwk3_datasets/yelp-valid.txt'
YELP_TEST_DATA = 'hwk3_datasets/yelp-test.txt'
IMDB_TRAIN_DATA = 'hwk3_datasets/IMDB-train.txt'
IMDB_VALID_DATA = 'hwk3_datasets/IMDB-valid.txt'
IMDB_TEST_DATA = 'hwk3_datasets/IMDB-test.txt'

# # Cleans the data by removing punctuations and making all letters lowercase
# def preprocess_text(file):
#     reviews, ratings = [], []
#     with open(file, 'r', encoding="utf-8") as f:
#         for l in f.readlines():
#             sp = l.split('\t')
#             ratings.append(int(sp[1].strip()))
#             reviews.append(re.compile('[^\w\s]').sub('', sp[0].strip()).lower())
#     return reviews, ratings
#
# # Counts the frequency of words in the reviews and returns the top 10000 words in descending order
# def count_freq_words(reviews):
#     count = Counter([word for line in reviews for word in line.split()]).most_common(10000)
#     words = [word[0] for word in count]
#     return words, count
#
# # Write the vocabulary data to a file
# def write_vocab_data(vocab_count, file):
#     f = open(file, 'w')
#     for index, key in enumerate(vocab_count):
#         f.write(str(key[1]) + ' ' + str(index + 1) + ' ' + str(key[0]) + '\n')
#     f.close()
#
# # Write vectors data to a file. Vectors indcate the indices of the non-zero values
# def write_vector_data(vectors, output, file):
#     f = open(file, 'w')
#     for i in range(vectors.shape[0]):
#         non_zero = np.flatnonzero(np.array(vectors[i])).tolist()
#         f.write((''.join([' ' + str(value) for value in non_zero])) + '\t' + str(output[i]) + '\n')
#     f.close()
#
#
# ''' Part 1 - Cleaning and Vectorizing Data \n '''
# print('Part 1 - Cleaning and Vectorizing Data \n')
# # Clean data
# yelp_train_rev, yelp_train_rat = preprocess_text(YELP_TRAIN_DATA)
# yelp_valid_rev, yelp_valid_rat = preprocess_text(YELP_VALID_DATA)
# yelp_test_rev, yelp_test_rat = preprocess_text(YELP_TEST_DATA)
# imdb_train_rev, imdb_train_rat = preprocess_text(IMDB_TRAIN_DATA)
# imdb_valid_rev, imdb_valid_rat = preprocess_text(IMDB_VALID_DATA)
# imdb_test_rev, imdb_test_rat = preprocess_text(IMDB_TEST_DATA)
#
# # Count the word frequencies
# yelp_vocab, yelp_count = count_freq_words(yelp_train_rev)
# imdb_vocab, imdb_count = count_freq_words(imdb_train_rev)
#
# # Write the top frequencies to a file
# write_vocab_data(yelp_count, 'generated_datasets/yelp-vocab.txt')
# write_vocab_data(imdb_count, 'generated_datasets/imdb-vocab.txt')
#
# # Create vectorizer object for vectorizing dataset
# vectorizer = CountVectorizer(vocabulary=yelp_vocab)
# vectorizer_bin = CountVectorizer(vocabulary=yelp_vocab, binary=True)
#
# # Create binary bag of words
# yelp_train_bin = vectorizer_bin.fit_transform(yelp_train_rev).todense()
# yelp_valid_bin = vectorizer_bin.fit_transform(yelp_valid_rev).todense()
# yelp_test_bin = vectorizer_bin.fit_transform(yelp_test_rev).todense()
# imdb_train_bin = vectorizer_bin.fit_transform(imdb_train_rev).todense()
# imdb_valid_bin = vectorizer_bin.fit_transform(imdb_valid_rev).todense()
# imdb_test_bin = vectorizer_bin.fit_transform(imdb_test_rev).todense()
#
# # Create frequency bag of words vectors
# yelp_train_freq = normalize(vectorizer.fit_transform(yelp_train_rev).todense())
# yelp_valid_freq = normalize(vectorizer.fit_transform(yelp_valid_rev).todense())
# yelp_test_freq = normalize(vectorizer.fit_transform(yelp_test_rev).todense())
# imdb_train_freq = normalize(vectorizer.fit_transform(imdb_train_rev).todense())
# imdb_valid_freq = normalize(vectorizer.fit_transform(imdb_valid_rev).todense())
# imdb_test_freq = normalize(vectorizer.fit_transform(imdb_test_rev).todense())
# write_vector_data(yelp_train_freq, yelp_train_rat, 'generated_datasets/yelp-train-vectors.txt')
# write_vector_data(yelp_valid_freq, yelp_valid_rat, 'generated_datasets/yelp-valid-vectors.txt')
# write_vector_data(yelp_test_freq, yelp_test_rat, 'generated_datasets/yelp-test-vectors.txt')
# write_vector_data(imdb_train_freq, imdb_train_rat, 'generated_datasets/imdb-train-vectors.txt')
# write_vector_data(imdb_valid_freq, imdb_valid_rat, 'generated_datasets/imdb-valid-vectors.txt')
# write_vector_data(imdb_test_freq, imdb_test_rat, 'generated_datasets/imdb-test-vectors.txt')
#
# print('All data cleaned and datasets vectorized. See generated_datasets folder for results.')
#
#
# def train_model(set, clf, params):
#     train = set['train']
#     valid = set['valid']
#     test = set['test']
#
#     train_input = train[0]
#     valid_input = valid[0]
#     test_input = test[0]
#
#     train_truth = train[1]
#     valid_truth = valid[1]
#     test_truth = test[1]
#
#     if params != None:
#         combine_input = sparse.vstack([train_input, valid_input])
#         combine_truth = np.concatenate((train_truth, valid_truth))
#         fold = [-1 for i in range(train_input.shape[0])] + [0 for i in range(valid_input.shape[0])]
#         ps = PredefinedSplit(test_fold=fold)
#         clf = GridSearchCV(clf, params, cv=ps, refit=True)
#         clf.fit(combine_input, combine_truth)
#     else:
#         clf.fit(train_input, train_truth)
#
#     best_param = None if params == None else clf.best_params_
#
#     f1_train = f1_score(train_truth, clf.predict(train_input), average=average)
#     f1_valid = f1_score(valid_truth, clf.predict(valid_input), average=average)
#     f1_test = f1_score(test_truth, clf.predict(test_input), average=average)
#
#     return f1_train, f1_valid, f1_test, best_param
#
# ''' Part 2 - Using Binary Bag Of Words Representation (Yelp) \n '''
# print('Part 2 - Using Binary Bag Of Words Representation (Yelp) \n')
#
# # Using random and majority classifiers
# random_class = DummyClassifier(strategy="uniform")
# majority_class = DummyClassifier(strategy="most_frequent")
# random_class.fit(yelp_train_bin, yelp_train_rat)
# majority_class.fit(yelp_train_bin, yelp_train_rat)
# random_class_prediction = random_class.predict(yelp_test_bin)
# majority_class_prediction = majority_class.predict(yelp_test_bin)
# print('Random Classifier Prediction: ', f1_score(yelp_test_rat, random_class_prediction, average='micro'))
# print('Majority Classifier Prediction: ', f1_score(yelp_test_rat, majority_class_prediction, average='micro'))
#
# # Using Naive Bayes Classifier
# param = [{'alpha': np.arange(0.6, 0.8, 0.01)}]
# pred = train_model(yelp_bow, BernoulliNB(), param)
# print(set, "Naive Bayes Classifier \n(train, valid, test) = ", pred[:3])
# print("best params = {}\n".format(pred[3]))

types = ['train.txt', 'valid.txt', 'test.txt', ]
ds_path = './hwk3_datasets/'
sets = ['yelp-', 'IMDB-']

def preprocess(file):
	"""
	Keyword arguments:
	file -- string file path (string)
	Returns:
	processed file (string)
	the function reads the file, puts everything in lower case and removes punctuation marks
	"""
	translator = str.maketrans(" ", " ", string.punctuation)
	with open(file, 'r', encoding="utf-8") as f:
		text = f.read()
	text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator)
	return text

def feature_extraction(name, n):
    """
    Keyword arguments:
    name -- set name (IMBD or yelp) - string
    n -- number of features - int
    Returns:
    dictionary {"words": ID}
    the function
    - extracts the top n frequent features with their respective ID.
    - writes output file for feature ID and count
    - writes output files for feature vectors for train, valid, test set
    """
    file = preprocess(ds_path + name + types[0])
    word_list = file.split(" ")
    counter = Counter(word_list).most_common(n)  # count the occurence of each word in the text
    dict = {}

    # save the top features in a output file "name-vocab.txt"
    # "word" id  count
    writer = open(name.split('-')[0] + '-vocab.txt', 'w')

    for i in range(n):
        word = counter[i][0]
        dict[word] = i + 1

        text = ("{}\t{}\t{}\n".format(word, i + 1, counter[i][1]))
        writer.write(text)

    # write feature vectors for every sample in train, valid and test sets
    for type in types:
        print(ds_path + name + type)
        file = preprocess(ds_path + name + type)

        examples = file.split("\n")[:-1]
        ds_output = [i[-1] for i in examples]

        writer = open(name.split('-')[0] + '-' + type.split('.')[0] + '.txt', 'w')
        for i in range(len(examples)):
            text = ""
            for word in examples[i].split(' ')[:-1]:
                if word in dict.keys():
                    text = "{} {}".format(text, dict[word])
            if len(text) == 0: text = ' '
            text = "{}\t{}\n".format(text, ds_output[i])
            writer.write(text[1:])

    return dict

set = sets[0]
vocab_list = feature_extraction(set, 10000)
print(vocab_list)
