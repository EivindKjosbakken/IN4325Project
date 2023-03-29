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


import os





app = Flask(__name__)
CORS(app)


APP_URL = "/"

print("opening tf idf matrix and corpus")

with open('IR/tfIdfMatrix.json') as json_file:
    tfIdfDict = json.load(json_file)
tfIdfMatrix = np.array(tfIdfDict["array"])

with open('IR/corpus.json') as json_file:
    corpus = json.load(json_file)
print("opened")


model=None


autocomplete_model = pipeline('text-generation', model='gpt2')

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
        indices = executeQuery(data["query"],model, tfIdfMatrix, corpus)
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
