from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from the file using pickle
with open('bodyfat.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from the form
    features = [
        float(request.form['density']),
        float(request.form['age']),
        float(request.form['weight']),
        float(request.form['height']),
        float(request.form['neck']),
        float(request.form['chest']),
        float(request.form['abdomen']),
        float(request.form['hip']),
        float(request.form['thigh']),
        float(request.form['knee']),
        float(request.form['ankle']),
        float(request.form['biceps']),
        float(request.form['forearm']),
        float(request.form['wrist']),
    ]
    
    # Make a prediction using the loaded model
    prediction = loaded_model.predict(np.array([features]))

    # Render the result
    return render_template('index.html', prediction=prediction[0])

# if __name__ == '__main__':
#     app.run(debug=True)
