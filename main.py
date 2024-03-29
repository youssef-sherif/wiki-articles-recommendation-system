from Document import Document
from Article import Article
from KMeansRecommend import KMeansRecommend
from gensim.models import Word2Vec, TfidfModel
from gensim.corpora import Dictionary
from pprint import pprint
from re import match
import numpy as np
import math
import random


def read_article_by_title(path, title):

    file = open(path, 'r')

    # Read the file line by line and exclude titles that are in the form ( = = = A title = = = )
    read_lines = False
    raw_article = ""
    lines = file.readlines()
    for line in lines:
        if read_lines:
            if match(r"([\s=\s]+)[\W+]+([\s=\s]+)", line):
                break
            else:
                raw_article += line

        if match(f"([\s=\s]+){title}([\s=\s]+)", line):
            read_lines = True
    file.close()

    return raw_article


def get_vector(corpus, model):
    vector = []
    for c in corpus:
        vector1 = []
        m = model[c]
        for i in range(0, 100):
            if i < len(m) - 1:
                vector1.append(m[i][1])
            else:
                vector1.append(0)
        vector.append(np.array(vector1))

    return np.array(vector)


def get_sum_vector(corpus, model):
    vector = []
    for c in corpus:
        m = model[c]
        sum = 0
        for n in m:
            sum += n[1]
        vector.append(np.array([sum]))

    return np.array(vector)


document = Document('./wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw')
document.pre_process()
document.build_n_grams(2)

vocab = Dictionary(document.n_grams)
corpus = [vocab.doc2bow(line) for line in document.n_grams]  # convert corpus to BoW format

model = TfidfModel(corpus)

vector = get_vector(corpus=corpus, model=model)

print(vector)

km = KMeansRecommend(data=vector)
km.k_means()

title = "Life"

raw = read_article_by_title('./wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw', title)
article = Article(title=title, raw=raw)

recommended_article_ids = km.recommend(article)

recommended_articles = []
for recommended_article_id in recommended_article_ids:
    recommended_articles.append(document.get_article(recommended_article_id))

print("= = RECOMMENDED ARTICLES = =")

for recommended_article in recommended_articles:
    print(recommended_article.title)
    print(recommended_article.raw)
