import nltk
import math
import numpy as np


def score(corpus, query):  # https://en.wikipedia.org/wiki/Okapi_BM25
    documents_token_frequencies = []
    D = []
    term_frequencies = {}
    num_doc = 0
    for document in corpus:
        doc_len = len(document)
        D.append(doc_len)
        num_doc += doc_len
        frequencies = {}
        for word in document:
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1
        documents_token_frequencies.append(frequencies)
        for word, frequency in frequencies.items():
            if word not in term_frequencies:
                term_frequencies[word] = 0
            term_frequencies[word] += 1
    corpus_len = len(corpus)
    avg_dl = num_doc / corpus_len
    idf = {}
    for term, frequency in term_frequencies.items():
        idf_term = math.log((corpus_len - frequency + 0.5) / (frequency + 0.5) + 1)
        idf[term] = idf_term
    k1 = 1.6
    b = 0.75
    score = np.zeros(corpus_len)
    for token in query:
        frequency = []
        for doc in documents_token_frequencies:
            frequency.append(doc.get(token, 0))
        frequency = np.array(frequency)
        score += (idf.get(token, 0)) * (frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * np.array(D) / avg_dl)))
    return list(score)


docs = []
for index, line in enumerate(open('Europarl.txt', 'r', encoding='utf8').readlines()):
    # if index % 100 == 99:
    docs.append(line.strip())

corpus = [[word.lower() for word in doc if word.isalpha()] for doc in [nltk.word_tokenize(doc) for doc in docs]]

while True:
    query = input('query: ')
    tokenized_query = [word.lower() for word in nltk.word_tokenize(query.strip()) if word.isalpha()]
    results = list(zip(score(corpus, tokenized_query), docs))
    pairs = sorted(results, key=lambda x: x[0], reverse=True)
    for pair in pairs[:10]:
        if pair[0] > 0:
            print(pair)

