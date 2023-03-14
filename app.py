from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    print("index page")
    return '350 projects'



@app.route('/predict_salary',methods=['POST'])
def predict():
    model = pickle.load(open('salary_prediction.pkl', 'rb'))

    print("Req form")
    print(request.form.values())

    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    data = {
        "status":'ok',
        "result":output
    }

    return jsonify(data)



@app.route('/predict_sentiment',methods=['POST'])
def sentiment_analysis():
    print(request.form)
    text=request.form['text']
    print(text)
    sentiment_score = TextBlob(text).sentiment.polarity
    data = {
        "status":"ok",
        "result":sentiment_score
    }
    return jsonify(data)