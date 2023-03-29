#imports
import pandas as pd
import numpy as np
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
from autocorrect import Speller
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
import time

#helper functions for indexing dataset
def getCleanedDf():
    dfRepo1 = pd.read_csv("../../data/tudelft_repository_1679269258.csv")
    dfRepo2 = pd.read_csv("../../data/tudelft_repository_1679269271.csv")
    dfRepo3 = pd.read_csv("../../data/tudelft_repository_1679269276.csv")
    dfRepo4 = pd.read_csv("../../data/tudelft_repository_1679269281.csv")
    dfRepo5 = pd.read_csv("../../data/tudelft_repository_1679269287.csv")
    dfRepo6 = pd.read_csv("../../data/tudelft_repository_1679269292.csv")

    dfRepo = pd.concat([dfRepo1, dfRepo2, dfRepo3, dfRepo4, dfRepo5, dfRepo6])
    df = dfRepo.dropna(subset=["abstract"]) #elements without abstract
    df = df.replace(np.nan, '', regex=True)

    return df

def createAndStoreTfIdfMatrix(corpus):

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdfMatrix = tfIdfVectorizer.fit_transform(corpus) #this is the matrix
    tfIdfMatrix =  tfIdfMatrix.toarray()

    tfIdfDict = {"array" : tfIdfMatrix.tolist()}

    with open("tfIdfMatrix.json", "w") as write_file:
        json.dump((tfIdfDict), write_file)
    
    return tfIdfVectorizer

def storeUniqueWordsDict(tfIdfVectorizer):
    uniqueWords = tfIdfVectorizer.get_feature_names_out() #unique words
    uniqueWordsIndexDict = dict()
    for idx, word in enumerate(uniqueWords):
        uniqueWordsIndexDict[word] = idx
        
    with open("uniqueWordsDict.json", "w") as write_file:
        json.dump(uniqueWordsIndexDict, write_file)
    
    return uniqueWords

def storeTokenizedCorpus(corpus, filename = "bm25_tokenized_corpus.json"):
    """stores tokenized corpus to be used for the BM25 model"""
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    with open(filename, "w") as write_file:
        json.dump(tokenized_corpus, write_file)

def createAndStoreIdfValues(corpus, uniqueWords):
    invDocFreq = dict.fromkeys(uniqueWords, 0)
    numDocuments = len(corpus)

    for text in corpus:

        tokens = set(text.split()) #unique
        for token in tokens: 
            if (token in invDocFreq):
                invDocFreq[token] += 1
    for key, value in invDocFreq.items():
        invDocFreq[key] = np.log(numDocuments/(value+1)) 

    avgDocFreq = np.mean(np.array(list(invDocFreq.values())))

    with open("idfDict.json", "w") as write_file:
        json.dump(invDocFreq, write_file)


#different preprocessing techniques -> make a preprocessed corpus to index on:
def removeStopwords(corpus : list):
    """remove stopwords from corpus"""
    nonStopwordCorpus = []
    for text in corpus:
        word_tokens = word_tokenize(text)
        nonStopwordCorpus.append(" ".join([w for w in word_tokens if not w.lower() in stop_words]))

    return (nonStopwordCorpus)

def allLowerCase(corpus : list):
    lowerCaseCorpus = []

    for text in corpus:
        lowerCaseCorpus.append(text.lower())

    return lowerCaseCorpus

def correctSpelling(corpus : list):
    spell = Speller(lang='en')
    correctedSpellingCorpus = []
    print("correcting spelling, can comment out if it takes a long time...")
    for text in tqdm(corpus):
        word_tokens = word_tokenize(text)
        correctedSpellingCorpus.append(" ".join([spell(w) for w in word_tokens]))

    return correctedSpellingCorpus

def stemming(corpus : list):
    """NOTE: should apply in the end since it often does not produce english words. 
    Applies stemming to all words in the corpus"""
    stemmer = PorterStemmer()
    stemmedCorpus = []
    for text in corpus:
        stemmedCorpus.append(' '.join(stemmer.stem(token) for token in nltk.word_tokenize(text)))

    return stemmedCorpus

def runPreprocessing(corpus : list):
    """Runs all preprocessing techniques. NOTE: order is important (atleast have stemming in the end!)"""
    corpus = removeStopwords(corpus)
    corpus = allLowerCase(corpus)
    # corpus = correctSpelling(corpus) #NOTE: this takes quite a while, can comment out if you want
    corpus = stemming(corpus)
    return corpus


def indexDataset(numberOfDocumentsToRank = 1000):
    #read data in

    df = getCleanedDf()

    corpusDict = dict()
  
    for idx, row in enumerate(df.to_numpy()): #NOTE: make sure corpus us df (and not specific col, is also used to return results)
        corpusDict[idx] = list(row)
        
    with open("corpus.json", "w") as write_file:
        json.dump(corpusDict, write_file)

    #get data
    abstractCorpus = df["abstract"].to_numpy()[:numberOfDocumentsToRank] 
    titleCorpus = df["title"].to_numpy()[:numberOfDocumentsToRank]

    #preprocess
    abstractCorpus = runPreprocessing(abstractCorpus)
    titleCorpus = runPreprocessing(titleCorpus)

    #implement tf-idf with sklearn:
    # tfIdfVectorizer = createAndStoreTfIdfMatrix(corpus) #NOTE removed now since we are using BM25 and not tfidf
    
    #store unique words dict
    # uniqueWords = storeUniqueWordsDict(tfIdfVectorizer)  #NOTE removed now since we are using BM25 and not tfidf

    # store IDF values:
    # createAndStoreIdfValues(corpus, uniqueWords) #NOTE removed now since we are using BM25 and not tfidf

    #store tokenized corpus_for_bm25:
    storeTokenizedCorpus(abstractCorpus, "bm25_tokenized_abstract_corpus.json")
    storeTokenizedCorpus(titleCorpus, "bm25_tokenized_title_corpus.json")






#order: remove stop words, all lower case, correct spelling, stemming
indexDataset(1000)
