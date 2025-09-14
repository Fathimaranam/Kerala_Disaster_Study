#•	“Run train_model.py first to generate model.pkl and scaler.pkl”
#•	“Place index.html inside a templates/ folder”

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        try:
            # Get data from form
            fatalities = float(request.form["fatalities"])
            camps = float(request.form["camps"])
            rainfall = float(request.form["rainfall"])
            normal_rainfall = float(request.form["normal_rainfall"])
            landslides = float(request.form["landslides"])
            damaged_houses = float(request.form["damaged_houses"])
            district_code = float(request.form["district_code"])  # 7th feature

            # Prepare input for model (7 features)
            X = np.array([[fatalities, camps, rainfall, normal_rainfall, landslides, damaged_houses, district_code]])
            X_scaled = scaler.transform(X)

            # Predict severity
            pred = model.predict(X_scaled)[0]
            prediction = str(pred)

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
