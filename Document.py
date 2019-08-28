from Article import Article
from math import log


class Document:

    def __init__(self, lines):
        """
            group articles together line by line until empty line is found.
            creates article objects and assigns an incremental id for each article.
        :param lines:
        """
        self.articles = []
        self.n_grams_vector = []

        raw_article = ""
        i = 0
        for line in lines:
            raw_article += line
            if line.isspace() and not raw_article.isspace():
                i += 1
                article = Article(raw_article, i)
                self.articles.append(article)
                raw_article = ""

    def pre_process(self):
        for article in self.articles:
            article.tokenize().remove_stop_words().lemmatize()

    def build_n_grams_dictionaries(self, n=2):
        """
            loops on all articles and builds dictionaries for each.
        :return: None
        """
        for article in self.articles:
            article.n_grams.append(article.get_n_grams(n))
            article.build_dictionary()

    def build_n_grams_vector(self, n=2):
        for article in self.articles:
            self.n_grams_vector.append(article.get_n_grams(n))

    def occurrence(self, word):
        count = 0
        for article in self.articles:
            for token in article.tokens:
                if word == token:
                    count += 1

        return count

    def idf(self, word):
        return log(self.articles.__len__() / self.occurrence(word), 2)

    def tf(self, word):
        """
            calculates article.tf(word) only if the article contains the word and creates a dictionary with article.id
            as key and tf as value.
        :param word:
        :return:
        """
        tf_results = {}

        for article in self.articles:
            if word in article.freq_dict:
                tf_results[article.id] = article.tf(word)

        return tf_results

    def tf_idf(self, word):
        """
            calculates article.tf(word) only if the article contains the word and creates a dictionary with article.id
            as key and tf * idf as value.
        :param word:
        :return:
        """

        tf_idf_results = {}
        idf_result = self.idf(word)

        for article in self.articles:
            if word in article.freq_dict:
                tf_idf_results[article.id] = article.tf(word) * idf_result

        return tf_idf_results
