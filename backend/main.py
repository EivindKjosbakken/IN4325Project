from flask import Flask, request
import requests
from flask_cors import CORS, cross_origin
from IR.tfIdf import executeQuery
import json
import numpy as np
import time

app = Flask(__name__)
CORS(app)


APP_URL = "/"

print("opening tf idf matrix and corpus")
with open('IR/abstractTfIdfMatrix.json') as json_file1: #TODO try storing as something better than json? 
        abstractTfIdfDict = json.load(json_file1)
abstractTfIdfMatrix = np.array(abstractTfIdfDict["array"])

with open('IR/titleTfIdfMatrix.json') as json_file2:
        titleTfIdfDict = json.load(json_file2)
titleTfIdfMatrix = np.array(titleTfIdfDict["array"])

with open('IR/corpus.json') as json_file:
    corpus = json.load(json_file)
print("opened")



@app.route(f"{APP_URL}/retrieve", methods=["POST"])
def retrieve():
    data = request.json  # if you want to retrieve data
    start = time.time()
    try:
        indices = executeQuery(data["query"], abstractTfIdfMatrix, titleTfIdfMatrix, corpus)
        return {"results" : (indices[:3]), "time": time.time() - start}, 200
    except Exception as e:
        print("cant execute query, error:", e)
        return "error", 400



@app.route(f"{APP_URL}/testGet", methods=["GET"])
def testGetApi():
    print("get test endpoint is called")

    isTest = True

    if isTest:
        return {"Test": "worked with get"}, 200
    return "Test api method failed", 400
