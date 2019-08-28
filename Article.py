from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter

wordnet_lemmatizer = WordNetLemmatizer()
english_stemmer = SnowballStemmer('english')


class Article:

    def __init__(self, raw, id):
        self.raw = raw
        self.id = id
        self.tokens = []
        self.n_grams = []
        self.freq_dict = {}

    def tokenize(self):
        """
            uses regex tokenizer and stores the values of tokens all in lower case.
        :return: self
        """
        tokenizer = RegexpTokenizer(r'\w+')
        self.raw = self.raw.lower()
        self.tokens = tokenizer.tokenize(self.raw)

        return self

    def remove_stop_words(self):
        stop_words = set(stopwords.words('english'))
        filtered_tokens = []
        for token in self.tokens:
            if token not in stop_words:
                filtered_tokens.append(token)

        self.tokens = filtered_tokens

        return self

    def stem(self):
        """
            use either stem or lemmatize but not both.
        :return: self
        """
        stemmed_words = []

        for word in self.tokens:
            stemmed_words.append(english_stemmer.stem(word))

        self.tokens = stemmed_words

        return self

    def lemmatize(self):
        """
            use either stem or lemmatize but not both.
        :return: self
        """
        lemmatized_words = []

        for word in self.tokens:
            lemmatized_words.append(wordnet_lemmatizer.lemmatize(word))

        self.tokens = lemmatized_words

        return self

    def build_dictionary(self):
        """
            builds a dictionary with article words as key and words occurrence in the article as value.
        :return: None as it updates freq_dict
        """

        for token in self.tokens:
            if token in self.freq_dict:
                self.freq_dict[token] += 1
            else:
                self.freq_dict[token] = 1

    def tf(self, word):
        return self.freq_dict[word] / self.freq_dict.__len__()

    def get_n_grams(self, n=2):
        n_grams = [' '.join(grams) for grams in ngrams(self.tokens, n)]
        n_gram_counts = Counter(n_grams)

        most_common_n_grams = n_gram_counts.most_common(10)

        n_grams = []
        for i in most_common_n_grams:
            if i[1] > 1:
                n_grams.append(i[0])

        return n_grams
