from Document import Document
import json
import re


def read_raw_data(path):
    lines = []

    file = open(path, 'r')

    # Read the file line by line and exclude titles that are in the form ( = = = A title = = = )
    for line in file.readlines():
        if not re.match(r"([\s=\s]+)[\W+]+([\s=\s]+)", line):
            lines.append(line)

    file.close()

    return lines


lines = read_raw_data('./wikitext-2-raw-v1/wikitext-2-raw/wiki.train.raw')
document = Document(lines)
document.pre_process()
document.build_dictionaries()

print("Some Examples: \n")

# example for a word that is very common in the document
print('\nwar')
print(document.tf('war'))
print(document.idf('war'))
print(document.tf_idf('war'))

# example for a word that is very rare in the document
print('\npâté')
print(document.tf('pâté'))
print(document.idf('pâté'))
print(document.tf_idf('pâté'))

# example for a word that is neither rare nor common in the document
print('\nvalkyria')
print(document.tf('valkyria'))
print(document.idf('valkyria'))
print(document.tf_idf('valkyria'))