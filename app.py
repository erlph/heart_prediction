from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model

model = pickle.load(open('models/classifier2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form inputs
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    hi = int(request.form['hi'])
    lo = int(request.form['lo'])
    cholesterol = int(request.form['cholesterol'])
    glucogen = int(request.form['glucogen'])
    smoke = int(request.form['smoke'])
    alcohol = int(request.form['alcohol'])
    activity = int(request.form['activity'])

    # Prepare the input data for prediction
    input_data = np.array([[gender, age, height, weight, hi, lo, cholesterol, glucogen, smoke, alcohol, activity]])

    # Make the prediction
    prediction = model.predict(input_data)[0]


    message_1 = "You have a high risk of having a cardiovascular condition. Please refer to a health professional as soon as possible."
    message_0 = "It appears you are in good health and do not seem to have a heart condition. I bid you a healthy life."
    if prediction==1:
        return render_template('index.html', prediction=message_1)
    else:
        return render_template('index.html', prediction=message_0)


    
    # Pass the prediction result to the template
    

if __name__ == '__main__':
    app.run(debug=True)
