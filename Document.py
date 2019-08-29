from Article import Article
from math import log
from re import match
from pprint import pprint


class Document:

    def __init__(self, path):
        """
            group articles together line by line until empty line is found.
        :param lines:
        """

        titles = {}

        file = open(path, 'r')

        # Read the file line by line and exclude titles that are in the form ( = = = A title = = = )
        current_title = ""
        lines = []
        for line in file.readlines():
            if match(r"([\s=\s]+)[\W+]+([\s=\s]+)", line):
                current_title = line
                lines = []
            else:
                lines.append(line)
                titles[current_title] = lines

        file.close()

        self.articles = []
        self.n_grams = []

        raw_article = ""
        i = 0
        for title in titles:
            for line in titles[title]:
                raw_article += line
                if line.isspace() and not raw_article.isspace():
                    article = Article(title, raw_article)
                    article.id = i
                    self.articles.append(article)
                    raw_article = ""
                    i += 1

    def pre_process(self):
        for article in self.articles:
            article.tokenize().remove_stop_words().lemmatize()

    def build_n_grams(self, n=2):
        for article in self.articles:
            self.n_grams.append(article.get_n_grams(n))

    def get_article(self, id):
        return self.articles[id]
