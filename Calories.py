from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('exercise form.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        print(request.form)
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heartbeat = float(request.form['heartbeat'])
        temperature = float(request.form['temperature'])



        # Arrange inputs into the expected order and shape
        features = np.array([[gender,age, height, weight, duration, heartbeat, temperature]])

        # Predict
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2><br><a href='/'>Go Back</a>"


if __name__ == '__main__':
    app.run(debug=True)