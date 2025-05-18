from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [float(request.form.get(feat)) for feat in ["pm2_5", "pm10", "no2", "so2", "co", "o3"]]
        pred = model.predict([features])[0]
        prediction = round(pred, 2)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
