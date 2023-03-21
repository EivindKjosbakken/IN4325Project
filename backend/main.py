from flask import Flask, request
import requests
from flask_cors import CORS, cross_origin

from IR.tfIdf import executeQuery


app = Flask(__name__)
CORS(app)


APP_URL = "/"


@app.route(f"{APP_URL}/retrieve", methods=["POST"])
def retrieve():
    data = request.json  # if you want to retrieve data
    # print(data["query"])
    try:
        indices = executeQuery("testing Python AI")
        return {"indices" : (indices[:3])}, 200
    except:
        print("cant execute query")
        return "error", 400


    isTest = True

    if isTest:
        return "Called test endpoint ", 200
    return "Test api method failed", 400


@app.route(f"{APP_URL}/testGet", methods=["GET"])
def testGetApi():
    print("get test endpoint is called")

    isTest = True

    if isTest:
        return {"Test": "worked with get"}, 200
    return "Test api method failed", 400
