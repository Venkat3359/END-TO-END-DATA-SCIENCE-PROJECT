import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']
data = pd.read_csv(url, header=None, names=columns)

# Train-test split
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)# END-TO-END-DATA-SCIENCE-PROJECT
    from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    pred = model.predict([data])[0]
    return render_template("index.html", prediction=f"Prediction: {'Diabetic' if pred else 'Non-Diabetic'}")

if __name__ == '__main__':
    app.run(debug=True)

    <!DOCTYPE html>
<html>
<head><title>Diabetes Predictor</title></head>
<body>
    <h2>Enter Patient Data</h2>
    <form action="/predict" method="post">
        {% for label in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age'] %}
        <p>{{label}}: <input type="text" name="{{label}}"></p>
        {% endfor %}
        <input type="submit" value="Predict">
    </form>
    <h3>{{ prediction }}</h3>
</body>
  </html>
    # Step 1: Train the model
python model_training.py

# Step 2: Start the web app
python app.py
http://127.0.0.1:5000
