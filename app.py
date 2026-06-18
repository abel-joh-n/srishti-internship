from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = float(request.form['age'])
    salary = float(request.form['salary'])

    data = np.array([[age, salary]])

    data = scaler.transform(data)

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    if prediction[0] == 1:
        confidence = round(probability[0][1] * 100, 2)
        result = f" Customer WILL Purchase ({confidence}%)"
    else:
        confidence = round(probability[0][0] * 100, 2)
        result = f" Customer WILL NOT Purchase ({confidence}%)"


    # INSERT HERE
    age_percent = min((age / 60) * 100, 100)
    salary_percent = min((salary / 150000) * 100, 100)

    return render_template(
        "index.html",
        prediction_text=result,
        age=age,
        salary=salary,
        confidence=confidence,
        age_percent=age_percent,
        salary_percent=salary_percent
    )

if __name__ == "__main__":
    app.run(debug=True)