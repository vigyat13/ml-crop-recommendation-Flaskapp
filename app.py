import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='templates')

# Load the trained model (trained with pandas DataFrame including 'season' as string or numeric encoding)
model = pickle.load(open("model.pkl", "rb"))

# Mapping for the season encoding (label encoding)
season_map = {
    'winter': 0,
    'summer': 1,
    'rainy': 2,
    'spring': 3
}

@app.route("/")
def index():
    return render_template("index.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    season = request.form['Season'].strip().lower()  # Clean season input

    # Check if the season is valid
    if season not in season_map:
        return render_template("index.html", prediction_text="❌ Invalid season. Choose from Winter, Summer, Rainy, or Spring.")

    # Convert season to its corresponding label encoding
    season_encoded = season_map[season]

    # Assign a fixed value for 'water availability' (adjust as per your logic)
    water_availability = 1  # You can change this depending on your application logic

    # Prepare the input data in the correct order of features (ensure the order matches model training)
    input_data = {
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'water availability': water_availability,
        'season': season_encoded  # Use encoded season (numeric value)
    }

    # Create DataFrame in the required order
    input_df = pd.DataFrame([input_data])

    # Predict the crop
    try:
        prediction = model.predict(input_df)[0]
    except ValueError as e:
        return render_template("index.html", prediction_text=f"❌ Error: {e}")

    return render_template("index.html", prediction_text=f"✅ Predicted Crop: {prediction}")


if __name__ == "__main__":
    app.run(debug=True)
