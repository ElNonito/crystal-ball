from flask import Flask
import run


app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello_world():
    text = "text"
    prediction = run.predict('python')
    return str(prediction)