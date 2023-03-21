#imports
import pandas as pd
import numpy as np
import numpy as np
from nltk.tokenize import  word_tokenize 
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json

def indexDataset(numberOfDocumentsToRank = 1000):
	#read data in
	dfRepo1 = pd.read_csv("../../data/tudelft_repository_1679269258.csv")
	dfRepo2 = pd.read_csv("../../data/tudelft_repository_1679269271.csv")
	dfRepo3 = pd.read_csv("../../data/tudelft_repository_1679269276.csv")
	dfRepo4 = pd.read_csv("../../data/tudelft_repository_1679269281.csv")
	dfRepo5 = pd.read_csv("../../data/tudelft_repository_1679269287.csv")
	dfRepo6 = pd.read_csv("../../data/tudelft_repository_1679269292.csv")

	dfRepo = pd.concat([dfRepo1, dfRepo2, dfRepo3, dfRepo4, dfRepo5, dfRepo6])
	dfWithAbstract = dfRepo.dropna(subset=["abstract"]) #elements without abstract

	corpusDict = dict()
	for idx, row in enumerate(dfWithAbstract.to_numpy()):
		corpusDict[idx] = list(row)

	with open("corpus.json", "w") as write_file:
		json.dump(corpusDict, write_file)



	#implement tf-idf:
	npWithAbstract = dfWithAbstract.to_numpy()[:numberOfDocumentsToRank, :] 

	#tf-idf with sklearn
	allAbstracts = npWithAbstract[:, 6]
	corpus = allAbstracts

	tfIdfVectorizer=TfidfVectorizer(use_idf=True)
	tfIdfMatrix = tfIdfVectorizer.fit_transform(corpus) #this is the matrix
	tfIdfMatrix =  tfIdfMatrix.toarray()

	# dataframe = pd.DataFrame(tfIdfMatrix) 
	# dataframe.to_csv("tfIdfMatrix.csv", header=False, index=False)
	tfIdfDict = {"array" : tfIdfMatrix.tolist()}

	with open("tfIdfMatrix.json", "w") as write_file:
		json.dump((tfIdfDict), write_file)
	

	uniqueWords = tfIdfVectorizer.get_feature_names_out() #unique words
	uniqueWordsIndexDict = dict()
	for idx, word in enumerate(uniqueWords):
		uniqueWordsIndexDict[word] = idx
		
	with open("uniqueWordsDict.json", "w") as write_file:
		json.dump(uniqueWordsIndexDict, write_file)


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



indexDataset(1000)