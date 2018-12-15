from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Welcom to the class DLB16HT201!!"
