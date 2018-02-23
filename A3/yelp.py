import re, string

YELP_TRAIN_DATA = 'hwk3_datasets/yelp-train.txt'
YELP_VALID_DATA = 'hwk3_datasets/yelp-valid.txt'
YELP_TEST_DATA = 'hwk3_datasets/yelp-test.txt'
IMDB_TRAIN_DATA = 'hwk3_datasets/IMDB-train.txt'
IMDB_VALID_DATA = 'hwk3_datasets/IMDB-valid.txt'
IMDB_TEST_DATA = 'hwk3_datasets/IMDB-test.txt'
NUM_FEATURES = 10000

# Preprocesses a text file by replacing removing all punctuations and making all text lowercase
def preprocess_text(file):
    with open(file, 'r', encoding="utf-8") as f:
        text = f.read()
    text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(str.maketrans(" ", " ", string.punctuation))
    return text

# Returns a dictionary of the most frequently used words in descending order
def count_freq_words(text):
    counts = dict()
    words = text.split()
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    sorted = [(counts[key], key) for key in counts]
    sorted.sort()
    sorted.reverse()
    return sorted[:10000]

# Writes the vocabulary data to a file
def write_vocab_data(vocab_count, file):
    f = open(file, 'w')
    for index, key in enumerate(vocab_count):
        f.write(str(key[1]) + ' ' + str(index+1) + ' ' + str(key[0]) + '\n')
    f.close()
    return

text = preprocess_text(YELP_TRAIN_DATA)
count = count_freq_words(text)
write_vocab_data(count, 'yelp-vocab.txt')

print(count)

