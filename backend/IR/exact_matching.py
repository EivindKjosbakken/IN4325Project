# imports
import pandas as pd
import numpy as np
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import time
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from rank_bm25 import BM25Okapi

stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import tensorflow_hub as hub


def preProcessQuery(query: str, model):
    # remove stopwords

    word_tokens = word_tokenize(query)
    query = (" ".join([w for w in word_tokens if not w.lower() in stop_words]))

    # all lower case
    query = query.lower()
    # spell check
    word_tokens = word_tokenize(query)

    spell = Speller(lang='en')
    query = (" ".join([spell(w) for w in word_tokens]))

    # stemming
    stemmer = PorterStemmer()
    query = (' '.join(stemmer.stem(token) for token in word_tokenize(query)))

    return query


def executeQuery(query: str, model, tfIdfMatrix=None, corpus=None, numberOfElementsToReturn=100, embedder=None,
                 goodQueries=None, corpusEmbedding=None):
    # first preprocess query same way dataset is preprocessed

    query = preProcessQuery(query, model)

    # do embedding after preprocessing, but before splitting the query stirng
    if (embedder):
        start = time.time()
        print("QUERY ER: ", query)
        queryEmbedding = np.array(embedder([query])).reshape(-1, 1)
        cosineSim = np.dot(corpusEmbedding, queryEmbedding) / (
                    np.linalg.norm(corpusEmbedding) * np.linalg.norm(queryEmbedding))
        idx = np.argmax(cosineSim)
        print("most sim sentence: ", goodQueries[idx])
        print("Comparing queries took: ", time.time() - start)

    query = query.split(" ")

    with open('IR/bm25_tokenized_abstract_corpus.json') as json_file1:  # TODO maybe load on startup of backend?
        tokenized_abstract_corpus = json.load(json_file1)
    with open('IR/bm25_tokenized_title_corpus.json') as json_file2:  # TODO maybe load on startup of backend?
        tokenized_title_corpus = json.load(json_file2)

    exact_matching_score=[0 for i in range(1000)]

    for query_term in query:
        print(query_term)
        for i,vi in enumerate(tokenized_title_corpus):
            for j in vi:
                if j==query_term:
                    exact_matching_score[i]+=1

    for query_term in query:
        for i,vi in enumerate(tokenized_abstract_corpus):
            for j in vi:
                if j==query_term:
                    exact_matching_score[i]+=1


    sortedIndices = np.argsort(exact_matching_score)[::-1]

    with open('exact_matching_first_titles.txt', 'a+') as f:
        for i in sortedIndices[:10]:
            f.write(corpus[str(i)][2])
            f.write('\n')
    # for i in sortedIndices[:10]:
    #     print(corpus

    orderedCorpusAccordingToQuery = []
    for idx in sortedIndices[:numberOfElementsToReturn]:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))

    return orderedCorpusAccordingToQuery





if __name__ == "__main__":
    results = executeQueryLocal(
        "AI in python with pytorch")  # results are a list of indices from most relvant to least relevant from the corpus
    # print(len(results), len(results[0]))
    results = np.array(results)
    print(results[:2, 6])

    # with open('corpus.json') as json_file:
    # corpus = json.load(json_file)
    # print("\n\n\n",corpus[str(int(855))])
    # results = executeQuery("AI in python with pytorch") #results are a list of indices from most relvant to least relevant from the corpus
    # print(results[0])

