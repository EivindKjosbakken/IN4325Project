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


def runPreprocessing(corpus : list):
    """NOTE: order is important (atleast have stemming in the end!)"""
    corpus = removeStopwords(corpus)
    corpus = allLowerCase(corpus)
    # corpus = correctSpelling(corpus) #NOTE: this takes quite a while, can comment out if you want
    corpus = stemming(corpus)
    return corpus



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

def createAndStoreTfIdfMatrix(corpus, filename = "tfIdfMatrix.json"):
    """make the tf idf matrix from the corpus, and saves it to file"""
    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdfMatrix = tfIdfVectorizer.fit_transform(corpus) #this is the matrix
    tfIdfMatrix =  tfIdfMatrix.toarray()

    tfIdfDict = {"array" : tfIdfMatrix.tolist()}

    with open(filename, "w") as write_file:
        json.dump((tfIdfDict), write_file)
    
    return tfIdfVectorizer

def storeUniqueWordsDict(tfIdfVectorizer, filename):
    uniqueWords = tfIdfVectorizer.get_feature_names_out() #unique words
    # uniqueWords = runPreprocessing(uniqueWords) #TODO this must also use preprocessing OR does it ? (unsure now), error with shapes if this is ran
    uniqueWordsIndexDict = dict()
    for idx, word in enumerate(uniqueWords):
        uniqueWordsIndexDict[word] = idx
        
    with open(filename, "w") as write_file:
        json.dump(uniqueWordsIndexDict, write_file)
    
    return uniqueWords

def createAndStoreIdfValues(corpus, uniqueWords, filename):
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

    with open(filename, "w") as write_file:
        json.dump(invDocFreq, write_file)

#different preprocessing techniques -> make a preprocessed corpus to index on:




def indexDataset(numberOfDocumentsToRank = 1000):
    #read data in
    print("Reading in data...")
    dfWithAbstract = getDfWithAbstracts()
    titleCorpus = dfWithAbstract["title"].to_numpy()[:numberOfDocumentsToRank] #NOTE: dont grab all elements for now

    corpusDict = dict() #store in dictionary for fast access
    for idx, row in enumerate(dfWithAbstract.to_numpy()):
        corpusDict[idx] = list(row)
    with open("corpus.json", "w") as write_file:
        json.dump(corpusDict, write_file)

    npWithAbstract = dfWithAbstract.to_numpy()[:numberOfDocumentsToRank, :] 
    abstractcCorpus = npWithAbstract[:, 6] #6th fol is the abstract

    #preprocess
    print("Preprocessing data...")
    abstractcCorpus = runPreprocessing(abstractcCorpus)
    titleCorpus = runPreprocessing(titleCorpus)

    #implement tf-idf with sklearn:
    print("Creating and storing TF-IDF matrices...") 
    start = time.time()
    abstractTfIdfVectorizer = createAndStoreTfIdfMatrix(abstractcCorpus, "abstractTfIdfMatrix.json")
    titleTfIdfVectorizer = createAndStoreTfIdfMatrix(titleCorpus, "titleTfIdfMatrix.json")
    print("Took: ", time.time() - start)
    #store unique words dict
    print("Storing unique words...")
    abstractUniqueWords = storeUniqueWordsDict(abstractTfIdfVectorizer, "abstractUniqueWordsDict.json")
    titleUniqueWords = storeUniqueWordsDict(titleTfIdfVectorizer, "titleUniqueWordsDict.json")

    # store IDF values:
    print("Creating and storing IDF values...")
    createAndStoreIdfValues(abstractcCorpus, abstractUniqueWords, "abstractIdfDict.json")
    createAndStoreIdfValues(titleCorpus, titleUniqueWords, "titleIdfDict.json")
    print("Done!")




if __name__ == "__main__":
    indexDataset(1000)
    # with open('abstractUniqueWordsDict.json') as json_file:
    #     abstractUniqueWordsIndexDict = json.load(json_file)

    # print(abstractUniqueWordsIndexDict.keys())