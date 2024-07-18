from flask import Flask, request, render_template
import joblib
import pandas as pd
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = {
        'Year': [request.form.get('year', type=int)],
        'manufactor': [request.form.get('manufacturer', default='Unknown')],
        'model': [request.form.get('model', default='Unknown')],
        'Hand': [request.form.get('hand', type=int)],
        'Gear': [request.form.get('gear', default='Unknown')],
        'capacity_Engine': [request.form.get('capacity_engine', type=float)],
        'Engine_type': [request.form.get('engine_type', default='Unknown')],
        'Prev_ownership': [request.form.get('prev_ownership', default='Unknown')],
        'Curr_ownership': [request.form.get('curr_ownership', default='Unknown')],
        'Area': [request.form.get('area', default='Unknown')],
        'City': [request.form.get('city', default='Unknown')],
        'Km': [request.form.get('km', type=float)],
        'Test': [request.form.get('test')],
        'Supply_score': [request.form.get('supply_score', type=float)],
        'Pic_num': [request.form.get('pic_num', type=int)],
        'Color': [request.form.get('color', default='Unknown')]
    }

    # Create DataFrame from the form data
    df = pd.DataFrame(data)

    # Process the data using the prepare_data function
    processed_data = prepare_data(df)

    # Predict the price using the trained model
    predicted_price = model.predict(processed_data)[0]

    # Render the home template with the predicted price
    return render_template('index.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)