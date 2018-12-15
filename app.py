import numpy as np
import pickle 
from flask import Flask, abort, jsonify, request
app = Flask(__name__)
def hello():
    return "Welcom to the class DLB16HT201!!"
load_model = pickle.load(open("../data/schools.csv","rb"))
@app.route('/predict', methods=['POST'])
def predict():
     # Error checking
     data = request.get_json(force=True)
     # Convert JSON to numpy array
     predict_request = [data['user_id'],data['nlid']]
     predict_request = np.array(predict_request)
     # Predict using the random forest model
     y = load_model.predict(predict_request)
     # Return prediction
     output = [y[0]]
     return jsonify(results=output)
