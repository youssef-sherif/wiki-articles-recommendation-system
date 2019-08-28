from Document import Document
from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from re import match
from pprint import pprint
from gensim import corpora
from nltk import cluster
from sklearn.cluster import KMeans
from nltk.cluster import euclidean_distance
from Kmeans import kmeans
import numpy as np
from pprint import pprint


def read_raw_data(path):
    lines = []

    file = open(path, 'r')

    # Read the file line by line and exclude titles that are in the form ( = = = A title = = = )
    for line in file.readlines():
        if not match(r"([\s=\s]+)[\W+]+([\s=\s]+)", line):
            lines.append(line)

    file.close()

    return lines


lines = read_raw_data('./wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw')
document = Document(lines)
document.pre_process()
document.build_n_grams_vector(2)

# print("Some Examples: \n")
#
# # example for a word that is very common in the document
# print('\nwar')
# print(document.tf('war'))
# print(document.idf('war'))
# print(document.tf_idf('war'))
#
# # example for a word that is very rare in the document
# print('\npâté')
# print(document.tf('pâté'))
# print(document.idf('pâté'))
# print(document.tf_idf('pâté'))
#
# # example for a word that is neither rare nor common in the document
# print('\nvalkyria')
# print(document.tf('valkyria'))
# print(document.idf('valkyria'))
# print(document.tf_idf('valkyria'))
#
#
# # Create CBOW model

# model = Word2Vec(document.n_grams_vector, min_count = 1, size = 300, window = 5)
# x = np.asmatrix(model.syn1neg)
# print(model.syn1neg.shape)
# print(x.shape)
dct = Dictionary(document.n_grams_vector)
corpus = [dct.doc2bow(line) for line in document.n_grams_vector]  # convert corpus to BoW format
model = TfidfModel(corpus)
vector = []
for i in corpus:
    # vector.append(model[i])
    x = model[i]
    z = 0
    for n in x:
        z += n[1]

    z = np.array([z])
    vector.append(z)
arr = np.array(vector)
print(arr.shape)
km = kmeans(arr)
new_centroids, clusters, error, iter_num = kmeans.kmeans(km.dataset)
for n in clusters:
    print(n)
# print(document.n_grams_vector)
# vectors = np.array(document.n_grams_vector)
#
# centroids = np.array([document.n_grams_vector[0], document.n_grams_vector[1], document.n_grams_vector[2]])
#
# k_means(3, vectors, centroids, model.n_similarity)
