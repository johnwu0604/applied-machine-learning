import re, string
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

YELP_TRAIN_DATA = 'hwk3_datasets/yelp-train.txt'
YELP_VALID_DATA = 'hwk3_datasets/yelp-valid.txt'
YELP_TEST_DATA = 'hwk3_datasets/yelp-test.txt'
IMDB_TRAIN_DATA = 'hwk3_datasets/IMDB-train.txt'
IMDB_VALID_DATA = 'hwk3_datasets/IMDB-valid.txt'
IMDB_TEST_DATA = 'hwk3_datasets/IMDB-test.txt'

# Cleans the data by removing punctuations and making all letters lowercase
def preprocess_text(file):
    reviews, ratings = [], []
    with open(file, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            sp = l.split('\t')
            ratings.append(int(sp[1].strip()))
            reviews.append(re.compile('[^\w\s]').sub('', sp[0].strip()).lower())
    return reviews, ratings

# Counts the frequency of words in the reviews and returns the top 10000 words in descending order
def count_freq_words(reviews):
    count = Counter([word for line in reviews for word in line.split()]).most_common(10000)
    words = [word[0] for word in count]
    return words, count

# Write the vocabulary data to a file
def write_vocab_data(vocab_count, file):
    f = open(file, 'w')
    for index, key in enumerate(vocab_count):
        f.write(str(key[1]) + ' ' + str(index + 1) + ' ' + str(key[0]) + '\n')
    f.close()

# Write vectors data to a file
def write_vector_data(vectors, output, file):
    f = open(file, 'w')
    for i in range(vectors.shape[0]):
        non_zero = np.flatnonzero(np.array(vectors[i])).tolist()
        f.write((''.join([' ' + str(item) for item in non_zero])) + '\t' + str(output[i]) + '\n')
    f.close()
    return


''' Part 1 - Cleaning and Vectorizing Data \n '''
# Clean data
yelp_train_rev, yelp_train_rat = preprocess_text(YELP_TRAIN_DATA)
yelp_valid_rev, yelp_valid_rat = preprocess_text(YELP_VALID_DATA)
yelp_test_rev, yelp_test_rat = preprocess_text(YELP_TEST_DATA)
imdb_train_rev, imdb_train_rat = preprocess_text(IMDB_TRAIN_DATA)
imdb_valid_rev, imdb_valid_rat = preprocess_text(IMDB_VALID_DATA)
imdb_test_rev, imdb_test_rat = preprocess_text(IMDB_TEST_DATA)

# Count the word frequencies
yelp_vocab, yelp_count = count_freq_words(yelp_train_rev)
imdb_vocab, imdb_count = count_freq_words(imdb_train_rev)

# Write the top frequencies to a file
write_vocab_data(yelp_count, 'generated_datasets/yelp-vocab.txt')
write_vocab_data(imdb_count, 'generated_datasets/imdb-vocab.txt')

# Create vectorizer object for vectorizing dataset
vectorizer = CountVectorizer(vocabulary=yelp_vocab)
vectorizer_bin = CountVectorizer(vocabulary=yelp_vocab, binary=True)
# Create frequency bag of words vectors
yelp_train_freq = normalize(vectorizer.fit_transform(yelp_train_rev).todense())
yelp_valid_freq = normalize(vectorizer.fit_transform(yelp_valid_rev).todense())
yelp_test_freq = normalize(vectorizer.fit_transform(yelp_test_rev).todense())
imdb_train_freq = normalize(vectorizer.fit_transform(imdb_train_rev).todense())
imdb_valid_freq = normalize(vectorizer.fit_transform(imdb_valid_rev).todense())
imdb_test_freq = normalize(vectorizer.fit_transform(imdb_test_rev).todense())
write_vector_data(yelp_train_freq, yelp_train_rat, 'generated_datasets/yelp-train-frequency-vectors.txt')
write_vector_data(yelp_valid_freq, yelp_valid_rat, 'generated_datasets/yelp-valid-frequency-vectors.txt')
write_vector_data(yelp_test_freq, yelp_test_rat, 'generated_datasets/yelp-test-frequency-vectors.txt')
write_vector_data(imdb_train_freq, yelp_train_rat, 'generated_datasets/imdb-train-frequency-vectors.txt')
write_vector_data(imdb_valid_freq, yelp_valid_rat, 'generated_datasets/imdb-valid-frequency-vectors.txt')
write_vector_data(imdb_test_freq, yelp_test_rat, 'generated_datasets/imdb-test-frequency-vectors.txt')
# Create binary bag of words
yelp_train_bin = vectorizer_bin.fit_transform(yelp_train_rev).todense()
yelp_valid_bin = vectorizer_bin.fit_transform(yelp_valid_rev).todense()
yelp_test_bin = vectorizer_bin.fit_transform(yelp_test_rev).todense()
imdb_train_bin = vectorizer_bin.fit_transform(imdb_train_rev).todense()
imdb_valid_bin = vectorizer_bin.fit_transform(imdb_valid_rev).todense()
imdb_test_bin = vectorizer_bin.fit_transform(imdb_test_rev).todense()
write_vector_data(yelp_train_bin, yelp_train_rat, 'generated_datasets/yelp-train-binary-vectors.txt')
write_vector_data(yelp_valid_bin, yelp_valid_rat, 'generated_datasets/yelp-valid-binary-vectors.txt')
write_vector_data(yelp_test_bin, yelp_test_rat, 'generated_datasets/yelp-test-binary-vectors.txt')
write_vector_data(imdb_train_bin, yelp_train_rat, 'generated_datasets/imdb-train-binary-vectors.txt')
write_vector_data(imdb_valid_bin, yelp_valid_rat, 'generated_datasets/imdb-valid-binary-vectors.txt')
write_vector_data(imdb_test_bin, yelp_test_rat, 'generated_datasets/imdb-test-binary-vectors.txt')
