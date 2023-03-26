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

def getDfWithAbstracts():
    dfRepo1 = pd.read_csv("../../data/tudelft_repository_1679269258.csv")
    dfRepo2 = pd.read_csv("../../data/tudelft_repository_1679269271.csv")
    dfRepo3 = pd.read_csv("../../data/tudelft_repository_1679269276.csv")
    dfRepo4 = pd.read_csv("../../data/tudelft_repository_1679269281.csv")
    dfRepo5 = pd.read_csv("../../data/tudelft_repository_1679269287.csv")
    dfRepo6 = pd.read_csv("../../data/tudelft_repository_1679269292.csv")

    dfRepo = pd.concat([dfRepo1, dfRepo2, dfRepo3, dfRepo4, dfRepo5, dfRepo6])
    dfWithAbstract = dfRepo.dropna(subset=["abstract"]) #elements without abstract
    dfWithAbstract = dfWithAbstract.replace(np.nan, '', regex=True)

    return dfWithAbstract

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


def storeTokenizedCorpus(corpus):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    with open("bm25_tokenized_corpus.json", "w") as write_file:
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



#stopwords (make the indexes from stopwords, but remember to return the actual abstract, not the stopword abstracts)
def removeStopwords(corpus : list):
    """remove stopwords from corpus"""
    nonStopwordCorpus = []
    for text in corpus:
        word_tokens = word_tokenize(text)
        nonStopwordCorpus.append(" ".join([w for w in word_tokens if not w.lower() in stop_words]))

    return (nonStopwordCorpus)

#lowercase all abstracts
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

#stemming
def stemming(corpus : list):
    """NOTE: should apply in the end since it often does not produce english words. 
    Applies stemming to all words in the corpus"""
    stemmer = PorterStemmer()
    stemmedCorpus = []
    for text in corpus:
        stemmedCorpus.append(' '.join(stemmer.stem(token) for token in nltk.word_tokenize(text)))

    return stemmedCorpus


def indexDataset(numberOfDocumentsToRank = 1000):
    #read data in
    dfWithAbstract = getDfWithAbstracts()

    corpusDict = dict()
    for idx, row in enumerate(dfWithAbstract.to_numpy()):
        corpusDict[idx] = list(row)

    with open("corpus.json", "w") as write_file:
        json.dump(corpusDict, write_file)


    npWithAbstract = dfWithAbstract.to_numpy()[:numberOfDocumentsToRank, :] 
    corpus = npWithAbstract[:, 6] #6th fol is the abstract
    #preprocess corpus, NOTE: order is important (atleast have stemming in the end!)
    start = time.time()
    corpus = removeStopwords(corpus)
    corpus = allLowerCase(corpus)
    # corpus = correctSpelling(corpus) #NOTE: this takes quite a while, can comment out if you want
    corpus = stemming(corpus)
    print("preprocessing took: ", time.time() - start)

    #implement tf-idf with sklearn:
    tfIdfVectorizer = createAndStoreTfIdfMatrix(corpus)
    
    #store unique words dict
    uniqueWords = storeUniqueWordsDict(tfIdfVectorizer)

    # store IDF values:
    createAndStoreIdfValues(corpus, uniqueWords)

    #store tokenized corpus_for_bm25:
    storeTokenizedCorpus(corpus)






#order: remove stop words, all lower case, correct spelling, stemming
indexDataset(1000)
