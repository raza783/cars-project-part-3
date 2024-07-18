#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import joblib
import pandas as pd
from car_data_prep import prepare_data

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route to handle form submissions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        'Year': [int(request.form['year'])],
        'manufactor': [request.form['manufacturer']],
        'model': [request.form['model']],
        'fuel': [request.form['fuel']],
        'transmission': [request.form['transmission']],
        'Test': [request.form['test']],
        'Gear': [request.form['gear']],
        'capacity_Engine': [request.form['capacity_engine']],
        'Prev_ownership': [request.form['prev_ownership']],
        'Curr_ownership': [request.form['curr_ownership']],
        'Area': [request.form['area']],
        'City': [request.form['city']],
        'Km': [request.form['km']],
        'Supply_score': [request.form['supply_score']],
        'Pic_num': [request.form['pic_num']],
        'Color': [request.form['color']]
    }

    # Create DataFrame from the form data
    df = pd.DataFrame(data)

    # Process the data using the prepare_data function
    processed_data = prepare_data(df)

    # Predict the price using the trained model
    predicted_price = model.predict(processed_data)[0]

    # Render the home template with the predicted price
    return render_template('index.html', predicted_price=predicted_price)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

