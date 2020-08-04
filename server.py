#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:51:54 2020

@author: samueladdotey
"""

import requests
import flask
from flask import request
from flask_cors import CORS, cross_origin
import sys
import pickle
import numpy as np

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# dictionary to interpret solutions for output   
solutions = {0: "SG G-Cash", 1:"SG MyHedge", 2:"SG Lyxor AP", 3:"SG Analyst", 4:"SG Coop", 5:"SG Clearing Solutions", 6:"SG D-View"}

app = flask.Flask(__name__)
app.config["DEBUG"] = False 
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources=r'/algo', headers='Content-Type')

    
@app.route('/algo',methods = ["GET"])
@cross_origin(origin='localhost', headers=['Content-Type', 'Authorization' ])

def index():
    response = requests.get("https://rzrwyrzidyxhyck.form.io/sam/submission")
    print('**************', file=sys.stderr)
    obj = response.json()
    info = { 
            "firstName": obj[0]['data']['fname'],
            "lastName": obj[0]['data']['lname'],
            "industry": obj[0]['data']['industry'],
            "kp": obj[0]['data']['keprocess']
    }
    
    myvals = np.array([info["industry"],info["kp"]]).reshape(1, -1)
    result = loaded_model.predict(myvals)
    
    prediction = solutions[(result[0])]
    print(prediction, file=sys.stderr)

    return flask.jsonify({
            "product": prediction,
             "firstName": info["firstName"],
             "lastName": info["lastName"]
            })


"""
@app.route('/evaluate', methods = ["POST"])

def predict():
    data = request.get_json()
    print(data, file=sys.stderr)
    return flask.jsonify(data)
"""

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 3001)
    
    
