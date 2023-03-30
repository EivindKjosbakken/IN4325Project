from flask import Flask, request
import requests
from flask_cors import CORS, cross_origin
#from IR.tfIdf import executeQuery
from IR.bm25 import executeQuery
import json
import numpy as np
import time
import gensim.downloader as api
import os
import traceback
from transformers import pipeline
import tensorflow_hub as hub


import os





app = Flask(__name__)
CORS(app)


APP_URL = "/"

print("opening tf idf matrix and corpus")

# with open('IR/tfIdfMatrix.json') as json_file: #NOTE commented out while not using tfidf
#     tfIdfDict = json.load(json_file)
# tfIdfMatrix = np.array(tfIdfDict["array"])

with open('IR/corpus.json') as json_file:
    corpus = json.load(json_file)
print("opened")


model=None

autocomplete_model = pipeline('text-generation', model='gpt2')

#load embedder and information it needs for query reformulation
embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
loaded_arrays = np.load('IR/goodQueries.npz', allow_pickle = True)
goodQueries = [np.array(loaded_arrays[file]) for file in loaded_arrays.files]
corpusEmbedding = np.load("IR/corpusEmbedding.npy")

# ###Comment out if not query expansion
# start = time.time()
# model = api.load('word2vec-google-news-300')
# print('model loaded')
# print(f'Time to load: {time.time()-start}')
# model.init_sims()
# print('initialized')
# ###comment out until here



@app.route(f"{APP_URL}/retrieve", methods=["POST"])
def retrieve():
    data = request.json  # if you want to retrieve data
    start = time.time()
    try:
        # indices = executeQuery(data["query"],model, tfIdfMatrix, corpus) #NOTE only for tfidf
        indices = executeQuery(query = data["query"],model = model, tfIdfMatrix = None, corpus = corpus, numberOfElementsToReturn=5, embedder=embedder, goodQueries=goodQueries, corpusEmbedding=corpusEmbedding)

        return {"results" : (indices[:3]), "time": time.time() - start}, 200
    except:
        print("cant execute query")
        return ("error: ", traceback.print_exc()), 400



@app.route(f"{APP_URL}/testGet", methods=["POST"])
def testGetApi():
    print("get test endpoint is called")

    isTest = True

    if isTest:
        return {"Test": "worked with get"}, 200
    return "Test api method failed", 400

@app.route(f"{APP_URL}/autocomplete", methods=["POST","GET"])
def autocomplete():
    if request.method == "GET":
        text = request.args.get('text')
        suggestions = autocomplete_model(text, max_length=30, num_return_sequences=3)
        return jsonify({'suggestions': suggestions})
