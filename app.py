import os
import pickle
from io import BytesIO

import requests
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        input_data = request.get_json()
        text = input_data["text"]
        platform = input_data["platform"]
        print(platform)
        vec = cv.transform([text]).toarray()
        result = model.predict(vec)
        if result[0] == 0:
            prediction_text = "Ham"
        else:
            prediction_text = "Spam"
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success predicting",
            },
            "data": {
                "prediction_text": prediction_text
            }
        })
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed",
            },
            "data": None,
        }), 405


@app.route("/media", methods=["GET", "POST"])
def media():
    if request.method == "POST":
        input_data = request.get_json()
        image_url = input_data["image_url"]
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        print(img)
        return jsonify({
            "status": {
                "code": 200,
                "message": "Success receive media",
            },
            "data": {
                "media": str(img),
                "type": str(type(img))
            }
        })
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed",
            },
            "data": None,
        }), 405


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
