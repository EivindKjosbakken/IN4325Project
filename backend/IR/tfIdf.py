#imports
import pandas as pd
import numpy as np
import numpy as np
from nltk.tokenize import  word_tokenize 
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import time
from autocorrect import Speller
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer



def preProcessQuery(query: str):
    #remove stopwords
    word_tokens = word_tokenize(query)
    query = (" ".join([w for w in word_tokens if not w.lower() in stop_words]))
    
    #all lower case
    query = query.lower()

    #spell check
    word_tokens = word_tokenize(query)
    spell = Speller(lang='en')
    query = (" ".join([spell(w) for w in word_tokens]))
    
    #stemming
    stemmer = PorterStemmer()
    query = (' '.join(stemmer.stem(token) for token in word_tokenize(query)))

    return query
    

# inference

def executeQuery(query: str, tfIdfMatrix = None, corpus = None, numberOfElementsToReturn = 5):

    #first preprocess query same way dataset is preprocessed
    query = preProcessQuery(query)

    with open('IR/idfDict.json') as json_file:
        invDocFreq = json.load(json_file)


    avgDocFreq = np.mean(np.array(list(invDocFreq.values())))
    uniqueWords = list(invDocFreq.keys())

    #make new vector of the input query
    Q = np.zeros(len(uniqueWords)) 
    tokens = query.split()
    counter = Counter(tokens)
    words_count = len(tokens)
    
    with open('IR/uniqueWordsDict.json') as json_file:
        uniqueWordsIndexDict = json.load(json_file)

    #calculate tf-idf scores for the input query
    for token in np.unique(tokens):
        if token not in uniqueWords: #cannot calc for word that does not exist
            print(f"word {token} does not exist in vocabulary")
            continue

        tf = counter[token]/words_count
        idf = invDocFreq.get(token, avgDocFreq)
        tfIdf = tf*idf
        #find idx of word in the vector
        idx = uniqueWordsIndexDict.get(token, None)
        if (idx is None):
            continue

        Q[idx] = tfIdf  

    #compare the input query vector to the vectors of all documents in corpus
    res = []

    for idx, doc in enumerate(tfIdfMatrix):

        cosineSim = np.dot(doc,Q)/(np.linalg.norm(doc)*np.linalg.norm(Q))
        res.append((idx, cosineSim))

    res = np.array(res)

    #sort the results
    res = res[res[:, 1].argsort()[::-1]]


    orderedCorpusAccordingToQuery = []
    for idx, cosineSim in res:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))
    return orderedCorpusAccordingToQuery[:numberOfElementsToReturn] 


def executeQueryLocal(query: str, numberOfElementsToReturn = 5):
    query = preProcessQuery(query)

    with open('idfDict.json') as json_file:
        invDocFreq = json.load(json_file)


    avgDocFreq = np.mean(np.array(list(invDocFreq.values())))
    uniqueWords = list(invDocFreq.keys())

    #make new vector of the input query
    Q = np.zeros(len(uniqueWords)) 
    tokens = query.split()
    counter = Counter(tokens)
    words_count = len(tokens)
    
    with open('uniqueWordsDict.json') as json_file:
        uniqueWordsIndexDict = json.load(json_file)

    #calculate tf-idf scores for the input query
    for token in np.unique(tokens):
        if token not in uniqueWords: #cannot calc for word that does not exist
            print(f"word {token} does not exist in vocabulary")
            continue

        tf = counter[token]/words_count
        idf = invDocFreq.get(token, avgDocFreq)
        tfIdf = tf*idf
        #find idx of word in the vector
        idx = uniqueWordsIndexDict.get(token, None)
        if (idx is None):
            continue

        Q[idx] = tfIdf  

    start = time.time()

    #compare the input query vector to the vectors of all documents in corpus


    with open('tfIdfMatrix.json') as json_file: 
        tfIdfDict = json.load(json_file)
    tfIdfMatrix = np.array(tfIdfDict["array"])
    tfIdfMatrix = np.array(tfIdfDict["array"])
    print("open tfidf matrix: ", time.time() - start)


    res = []

    start = time.time()
    for idx, doc in enumerate(tfIdfMatrix):

        cosineSim = np.dot(doc,Q)/((np.linalg.norm(doc)*np.linalg.norm(Q))+1) # +1 so we don't divide by 0
        res.append((idx, cosineSim))

    res = np.array(res)

    #sort the results
    res = res[res[:, 1].argsort()[::-1]]

    # if we want to return the actualy abstracts, then return this, else is returns the indices
    with open('corpus.json') as json_file:
        corpus = json.load(json_file)

    orderedCorpusAccordingToQuery = []
    for idx, cosineSim in res:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))
    
    return orderedCorpusAccordingToQuery[:numberOfElementsToReturn] 



if __name__ == "__main__":
    results = executeQueryLocal("AI in python with pytorch") #results are a list of indices from most relvant to least relevant from the corpus
    # print(len(results), len(results[0]))
    results = np.array(results)
    print(results[:2, 6])

    # with open('corpus.json') as json_file:
        # corpus = json.load(json_file)
    # print("\n\n\n",corpus[str(int(855))])
    # results = executeQuery("AI in python with pytorch") #results are a list of indices from most relvant to least relevant from the corpus
    # print(results[0])

