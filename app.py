import numpy as np
import pickle 
from flask import Flask, abort, jsonify, request
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcom to the class DLB16HT201!!"
