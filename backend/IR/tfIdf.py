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





# inference

def executeQuery(query: str):
    query = query.lower()
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
    tfIdfMatrix = np.genfromtxt('IR/tfIdfMatrix.csv', delimiter=',')
    res = []

    for idx, doc in enumerate(tfIdfMatrix):

        cosineSim = np.dot(doc,Q)/(np.linalg.norm(doc)*np.linalg.norm(Q))
        res.append((idx, cosineSim))

    res = np.array(res)

    #sort the results
    res = res[res[:, 1].argsort()[::-1]]

    # if we want to return the actualy abstracts, then return this, else is returns the indices
    with open('IR/corpus.json') as json_file:
        corpus = json.load(json_file)

    # """
    orderedCorpusAccordingToQuery = []
    for idx, cosineSim in res:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))
    return orderedCorpusAccordingToQuery[:5] #TODO only returning the first 5
    # """
    #only return the indices
    return np.array(res)[:, 0]  



# results = executeQuery("AI in python with pytorch") #results are a list of indices from most relvant to least relevant from the corpus
# print(results[0])
"""

numberOfDocumentsToShow = 3
for i in range(numberOfDocumentsToShow):
    print(dfWithAbstract.iloc[int(results[i]), 6])

"""