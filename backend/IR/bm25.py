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
from rank_bm25 import BM25Okapi


stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer


def preProcessQuery(query: str):
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


# inference

def executeQuery(query: str, tfIdfMatrix=None, corpus=None, numberOfElementsToReturn=5):
    # first preprocess query same way dataset is preprocessed
    query = preProcessQuery(query)
    with open('IR/bm25_tokenized_corpus.json') as json_file:
        tokenized_corpus = json.load(json_file)


    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores=bm25.get_scores(query)
    sortedIndices = np.argsort(doc_scores)[::-1]


    orderedCorpusAccordingToQuery = []
    for idx in sortedIndices[:numberOfElementsToReturn]:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))

    return orderedCorpusAccordingToQuery

        # print("time to get query vector ", time.time() - start2) #non relevant
    # compare the input query vector to the vectors of all documents in corpus

    # vectorized version
    # cosineSim = np.dot(tfIdfMatrix, Q) / (np.linalg.norm(tfIdfMatrix) * np.linalg.norm(Q))
    # sortedIndices = np.argsort(cosineSim)[::-1]  # reverse to have highest cosine similarity first

    #orderedCorpusAccordingToQuery = []
    #for idx in sortedIndices[:numberOfElementsToReturn]:
    #    orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))




def executeQueryLocal(query: str, numberOfElementsToReturn=5):
    # first preprocess query same way dataset is preprocessed
    query = preProcessQuery(query)
    query = query.split(" ")
    with open('bm25_tokenized_corpus.json') as json_file:
        tokenized_corpus = json.load(json_file)

    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(query)
    sortedIndices = np.argsort(doc_scores)[::-1]

    with open('corpus.json') as json_file:
        corpus = json.load(json_file)

    orderedCorpusAccordingToQuery = []
    for idx in sortedIndices[:numberOfElementsToReturn]:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))

    return orderedCorpusAccordingToQuery

    # compare the input query vector to the vectors of all documents in corpus



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

