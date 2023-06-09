from flask import Flask, request, jsonify
import requests
from flask_cors import CORS, cross_origin
#from IR.tfIdf import executeQuery
from IR.bm25 import executeQuery as executeQueryBM
from IR.exact_matching import executeQuery as executeQueryEM
import json
import numpy as np
import time
import gensim.downloader as api
import os
import traceback
from transformers import pipeline
import tensorflow_hub as hub
import rbo
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


# Data index meanings
# 0: uuid, 1: repository link, 2: title, 3: author, 4: contributor, 5: publication year, 6: abstract, 7: subject topic,
# 8: language, 9: publication type, 10: publisher, 11: isbn, 12: issn, 13: patent, 14: patent status, 15: bibliographic note,
# 16: access restriction, 17: embargo date, 18: faculty, 19: department, 20: research group, 21: programme, 22: project, 23: coordinates
def executeFilter(indices, filters):
    getIndex = { 0: "uuid", 1: "repository link", 2: "title", 3: "author", 4: "contributor", 5: "publication year", 6: "abstract", 7: "subject topic",
    8: "language", 9: "publication type", 10: "publisher", 11: "isbn", 12: "issn", 13: "patent", 14: "patent status", 15: "bibliographic note",
    16: "access restriction", 17: "embargo date", 18: "faculty", 19: "department", 20: "research group", 21: "programme", 22: "project", 23: "coordinates" }
    for filter in filters:
        if len(filters[filter]) == 0:
            continue
        index = [k for k, v in getIndex.items() if v == filter]
        for element in indices[:]:  # Iterate over a copy of the original list
            for f in filters[filter]:
                if f in element[index[0]]:
                    break
            else:
                indices.remove(element)
    return indices

@app.route(f"{APP_URL}/runmany", methods=["GET"])
def runMany():
    # with open("./IR/queriesToRun.txt", encoding='utf-8') as file:
    #     array = file.readlines()
    # print(array)
    text_file = open("./IR/queriesToRun.txt", "r", encoding="utf-8", errors='ignore')
    queries = text_file.read().split('\n')
    scores = []
    compare = True
    for query in queries:
        start = time.time()
        indices = executeQueryBM(query = query ,model = model, tfIdfMatrix = None, corpus = corpus, numberOfElementsToReturn=5, embedder=embedder, goodQueries=goodQueries, corpusEmbedding=corpusEmbedding)
        if compare:
            indicesCompare = executeQueryEM(query = query,model = model, tfIdfMatrix = None, corpus = corpus, numberOfElementsToReturn=5, embedder=embedder, goodQueries=goodQueries, corpusEmbedding=corpusEmbedding)
            indicesBMIds = [item[0] for item in indices]
            indicesEMIds = [item[0] for item in indicesCompare]
            compareScore = rbo.RankingSimilarity(indicesBMIds, indicesEMIds).rbo(p = 0.9)
            scores.append({'score' : compareScore, 'time': time.time() - start})
    avgScore = sum([i['score'] for i in scores]) / len(scores)
    avgtime = sum([i['time'] for i in scores]) / len(scores)
    print(scores)
    print(avgScore)
    return {"scores": scores, "avgScore": avgScore, "avgtime": avgtime}

@app.route(f"{APP_URL}/retrieve", methods=["POST"])
def retrieve():
    data = request.json  # if you want to retrieve data
    compareScore = -1
    start = time.time()
    try:
        # indices = executeQuery(data["query"],model, tfIdfMatrix, corpus) #NOTE only for tfidf
        indices = executeQueryBM(query = data["query"],model = model, tfIdfMatrix = None, corpus = corpus, numberOfElementsToReturn=5, embedder=embedder, goodQueries=goodQueries, corpusEmbedding=corpusEmbedding)
        if data["compare"]:
            indicesCompare = executeQueryEM(query = data["query"],model = model, tfIdfMatrix = None, corpus = corpus, numberOfElementsToReturn=5, embedder=embedder, goodQueries=goodQueries, corpusEmbedding=corpusEmbedding)
            indicesBMIds = [item[0] for item in indices]
            indicesEMIds = [item[0] for item in indicesCompare]
            compareScore = rbo.RankingSimilarity(indicesBMIds, indicesEMIds).rbo(p = 0.9)

        indices = executeFilter(indices, data["filters"])
        # print(data)
        filters = data.get("filters", None)
        if (filters):
            indices = executeFilter(indices, filters)

        return {"results" : (indices[:10]), "time": time.time() - start, "compareScore": compareScore }, 200
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
    if request.method == "POST":
        data = request.json  # if you want to retrieve data
        suggestions = autocomplete_model(data["query"], max_length=30, num_return_sequences=3)
        print("=========")
        print(suggestions)
        # broken for now
        return jsonify({suggestions})

    if request.method == "GET":
        text = request.args.get('text')
        suggestions = autocomplete_model(text, max_length=30, num_return_sequences=3)
        return jsonify({'suggestions': suggestions})
