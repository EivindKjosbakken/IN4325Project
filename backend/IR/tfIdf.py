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

def executeQuery(query: str, abstractTfIdfMatrix = None, titleTfIdfMatrix = None, corpus = None, numberOfElementsToReturn = 5):
    #first preprocess query same way dataset is preprocessed
    query = preProcessQuery(query)
    tokens = query.split()
    counter = Counter(tokens)
    words_count = len(tokens)
    #open abstract files
    with open('IR/abstractIdfDict.json') as json_file:
        abstractInvDocFreq = json.load(json_file)
    abstractAvgDocFreq = np.mean(np.array(list(abstractInvDocFreq.values())))
    abstractUniqueWords = list(abstractInvDocFreq.keys())
    with open('IR/abstractUniqueWordsDict.json') as json_file:
        abstractUniqueWordsIndexDict = json.load(json_file)

    abstractQ = np.zeros(len(abstractUniqueWords))

    #open title files
    with open('IR/titleIdfDict.json') as json_file:
        titleInvDocFreq = json.load(json_file)
    titleAvgDocFreq = np.mean(np.array(list(titleInvDocFreq.values())))
    titleUniqueWords = list(titleInvDocFreq.keys())
    with open('IR/titleUniqueWordsDict.json') as json_file:
        titleUniqueWordsIndexDict = json.load(json_file)

    titleQ = np.zeros(len(titleUniqueWords))
    #make new vector of the input query

    #calculate tf-idf scores for the input query
    for token in np.unique(tokens):
        #compare abstract
        if token in abstractUniqueWordsIndexDict: #cannot calc for word that does not exist
            tf = counter[token]/words_count
            idf = abstractInvDocFreq.get(token, abstractAvgDocFreq)
            tfIdf = tf*idf
            #find idx of word in the vector
            idx = abstractUniqueWordsIndexDict.get(token, None)
            if (idx is None):
                print("abstract continuing")
                continue
            abstractQ[idx] = tfIdf      
        else:
            print(f"word {token} does not exist in abstract vocabulary")

        #compare title
        if token in titleUniqueWordsIndexDict: #cannot calc for word that does not exist
            tf = counter[token]/words_count
            idf = titleInvDocFreq.get(token, titleAvgDocFreq)
            tfIdf = tf*idf
            #find idx of word in the vector
            idx = titleUniqueWordsIndexDict.get(token, None)
            if (idx is None):
                print("title continuing")
                continue
            titleQ[idx] = tfIdf      
        else:
            print(f"word {token} does not exist in title vocabulary")



    #compare the input query vector to the vectors of all documents in corpus

    #cosine sim for abstract
    abstractWeight = 0.3 #NOTE: defining importance of feature here
    abstractCosineSim = (np.dot(abstractTfIdfMatrix,abstractQ)/(np.linalg.norm(abstractTfIdfMatrix)*np.linalg.norm(abstractQ)))*abstractWeight

    #cosine sim for title
    titleWeight = 0.7
    titleCosineSim = (np.dot(titleTfIdfMatrix,titleQ)/(np.linalg.norm(titleTfIdfMatrix)*np.linalg.norm(titleQ)))*titleWeight

    totalCosineSim = (abstractCosineSim + titleCosineSim) #pluss together each element in arrays

    sortedIndices = np.argsort(totalCosineSim)[::-1] #reverse to have highest cosine similarity first
    print(sortedIndices[:5])
    orderedCorpusAccordingToQuery = []
    for idx in sortedIndices[:numberOfElementsToReturn]:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))
    
    return orderedCorpusAccordingToQuery


def executeQueryLocal(query: str, numberOfElementsToReturn = 5):
    query = preProcessQuery(query)

    with open('abstractIdfDict.json') as json_file:
        abstractInvDocFreq = json.load(json_file)
    abstractAvgDocFreq = np.mean(np.array(list(abstractInvDocFreq.values())))
    abstractUniqueWords = list(abstractInvDocFreq.keys())

    with open('titleIdfDict.json') as json_file:
        titleInvDocFreq = json.load(json_file)
    titleAvgDocFreq = np.mean(np.array(list(titleInvDocFreq.values())))
    titleUniqueWords = list(titleInvDocFreq.keys())



    #make new vector of the input query
    abstractQ = np.zeros(len(abstractUniqueWords)) 
    tokens = query.split()
    counter = Counter(tokens)
    words_count = len(tokens)
    
    with open('abstractUniqueWordsDict.json.json') as json_file:
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


    with open('abstractTfIdfMatrix.json') as json_file: 
        tfIdfDict = json.load(json_file)
    tfIdfMatrix = np.array(tfIdfDict["array"])
    tfIdfMatrix = np.array(tfIdfDict["array"])

    # if we want to return the actualy abstracts, then return this, else is returns the indices
    with open('corpus.json') as json_file:
        corpus = json.load(json_file)

    #vectorized version
    cosineSim = np.dot(tfIdfMatrix,Q)/(np.linalg.norm(tfIdfMatrix)*np.linalg.norm(Q))
    sortedIndices = np.argsort(cosineSim)[::-1] #reverse to have highest cosine similarity first
    orderedCorpusAccordingToQuery = []
    for idx in sortedIndices[:numberOfElementsToReturn]:
        orderedCorpusAccordingToQuery.append((corpus[str(int(idx))]))
    
    return orderedCorpusAccordingToQuery



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

