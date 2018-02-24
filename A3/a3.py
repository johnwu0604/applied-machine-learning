import re, string
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

num_features = 10000
ds_path = './hwk3_datasets/'
ds_types = ['train', 'valid', 'test']
ds = ['yelp-', 'IMDB-']
average = 'micro'

def extract_features(set, n):
    file = ds_path + set + ds_types[0] + '.txt'
    reviews = []
    # clean text by deleting punctuation and converting to all lower case
    with open(file, 'r', encoding="utf-8") as f:
        for l in f.readlines():
            sp = l.split('\t')
            reviews.append(re.compile('[^\w\s]').sub('', sp[0].strip()).lower())
    # find top features
    words = Counter([word for line in reviews for word in line.split()]).most_common(num_features)
    dict = {}
    for i in range(num_features):
        word = words[i][0]
        dict[word] = i + 1
    return dict

def to_bow(dict, set):
	binary = {}
	freq = {}
	for type in ds_types:
            file = ds_path + set + type + '.txt'
            translator = str.maketrans(" ", " ", string.punctuation)
            with open(file, 'r', encoding="utf-8") as f:
                text = f.read()
            text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator).split('\n')
            text = list(filter(None, text))
            output = np.array([int(line[-1]) for line in text])
            examples = [line[:-1] for line in text]
            vectorizer = CountVectorizer(vocabulary=dict.keys())
            vectors = np.asarray(vectorizer.fit_transform(examples).todense())
            #save freq and binary as sparse vectors for faster training
            freq = sparse.csr_matrix(normalize(vectors))
            vectors[vectors > 1] = 1 #set all count > 1 to 1, to binarize the vector
            binary = sparse.csr_matrix(vectors)
            binary[type] = [binary, output]
            freq[type] = [freq, output]
	return binary, freq

''' Part 1 - Cleaning and Vectorizing Data \n '''
print('Part 1 - Cleaning and Vectorizing Data \n')

features = extract_features(ds[0], num_features)
bin, freq = to_bow(features, ds[0])
print(freq)