#!/usr/bin/env python

from flask import Flask, render_template, request, jsonify
import numpy as np
from transformers import pipeline
import json

classifier = pipeline("sentiment-analysis")

generator=pipeline("text-generation",model="distilgpt2")
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

@app.route("/")
def web():
    return render_template("web.html")

@app.route(
    "/Prediction1", methods=["POST"])  # this line will be activated when Submit button has been pressed
def Prediction1():

    height_strings = request.form.get("input1")
    res1=classifier(height_strings)


    return render_template(
        "web.html", result="Based on the sentiment analysis your sentence is {} with the socre of {: .1f} out of 1.".format(res1[0].get('label'), res1[0].get('score'))
    )

@app.route(
    "/Prediction2", methods=["POST"])  # this line will be activated when Submit button has been pressed
def Prediction2():

    height_strings2 = request.form.get("input2")
    #res2=classifier(height_strings2)
    res2=generator(height_strings2,max_length=30,num_return_sequences=2,)


    return render_template(               
"web.html", result="Based on your input I generated: {} ".format(res2[1]['generated_text'])
    )

@app.route(
    "/status.json", methods=["GET"]
)
def PredictionStatus():
    current_prediction = [
        {
            "user_message": "Hello, my name is Behrooz",
            "model_response": "Hello, Behrooz! How are you today?"
        },
        {
            "user_message": "I'm doing well today! How are you?",
            "model_response": "I'm really enjoying our weather today!"
        }
    ]
    
    return json.dumps(current_prediction)


if __name__ == "__main__":
    app.run(debug=True)
