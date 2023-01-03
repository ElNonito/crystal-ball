from flask import Flask
from run import predict

app = Flask(__name__)

@app.route('/', methods=["GET"])
def hello_world():
    text = """
    This is the welocome page of my app.
    To make a prediction use the route /predict/input_your_sentence_with_underscores_for_space
    
    Somehow I miss-trained the model (probabilties where not between 1 and 0 but -10/10) so the predictions are mostly always the same.
    """
    prediction = predict('python')
    return str(prediction)


@app.route('/predict/<sentence>', methods=["GET"])
def predict_sentence(sentence):
    sentence.replace(" ", "_")
    prediction = predict(sentence)
    return str(f'Prediction for :{sentence} \nIs :{prediction}')


if __name__ == "__main__":
    app.run()
    print('End of service')
