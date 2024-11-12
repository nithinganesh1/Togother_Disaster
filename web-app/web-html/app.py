from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the trained models and scalers at the start of the app
rainfall_model = load_model('Rainfall/rain_model.keras') 
el_nino_model = load_model('el-nina/my_model.keras')  

# Load the landslide model
with open('landslide/model.pkl', 'rb') as file:
    landslide_model = pickle.load(file)

scaler = MinMaxScaler(feature_range=(0, 1))

# Load the last 60 rainfall data because LSTM needs the last trained data for prediction
last_60_data = pd.read_csv('Rainfall/last_60_rainfall_data.csv', index_col=0, parse_dates=True)
scaler.fit(last_60_data.values)  # Fit the scaler on the last 60 data

def predict_rainfall_with_year_month(model, scaler, last_data, year, month):
    max_year = 2050
    min_year = 2006

    # Normalize the inputs
    year_norm = (year - min_year) / (max_year - min_year)
    month_norm = month / 12

    # Create the input sequence using the last 60 data and append the normalized year and month
    last_scaled = scaler.transform(last_data.values)  # Scale the last data
    input_sequence = np.append(last_scaled, [[year_norm], [month_norm]], axis=0)

    # Reshape the input for prediction (60 time steps)
    input_sequence = input_sequence[-60:].reshape(1, 60, 1)  # Take the last 60 time steps
    predicted_scaled = model.predict(input_sequence)

    # Inverse transform to get the predicted rainfall value
    predicted_value = scaler.inverse_transform(predicted_scaled)[0][0]

    return round(predicted_value, 2)

# Function to predict El-Niño/La-Niña
def predict_el_nino(target_year, last_trained_year=2017, last_trained_month=12):
    input_data = np.zeros((1, 60, 1))  # Initialize input data
    initial_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example known values
    input_data[0, -len(initial_values):, 0] = initial_values

    current_year = last_trained_year
    current_month = last_trained_month
    predictions = []

    while current_year < target_year or (current_year == target_year and current_month < 12):
        predicted_scaled_value = el_nino_model.predict(input_data, verbose=0)
        predicted_scaled_value = np.reshape(predicted_scaled_value, (1, 1, 1))
        predictions.append(predicted_scaled_value[0, 0, 0])
        input_data = np.append(input_data[:, 1:, :], predicted_scaled_value, axis=1)

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    predicted_value = predictions[-1]
    return predicted_value


def predict_landslide(height, slope, rainfall):
    test = pd.DataFrame({
        'height': [height],
        'slope': [slope],
        'rainfall': [rainfall]
    })
    
    prediction = landslide_model.predict(test)
    if prediction[0] == 0:
        return "No landslide risks found"
    else:
        return "Landslide risks detected"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('service.html')

@app.route('/weather')
def weather_page():
    return render_template('weather.html')

@app.route('/heat', methods=['GET', 'POST'])
def heat():
    predicted_value = None
    year = None
    if request.method == 'POST':
        year = int(request.form['year'])
        predicted_value = predict_el_nino(year)

    return render_template('heat.html', predicted_value=predicted_value, year=year)

@app.route('/rainfall', methods=['GET', 'POST'])
def rainfall():
    predicted_rainfall = None
    if request.method == 'POST':
        year = int(request.form['year'])
        month = int(request.form['month'])
        predicted_rainfall = predict_rainfall_with_year_month(rainfall_model, scaler, last_60_data, year, month)

    return render_template('rainfall.html', predicted_rainfall=predicted_rainfall)

@app.route('/el-nino', methods=['GET', 'POST'])
def el_nino():
    predicted_el_nino = None
    if request.method == 'POST':
        year = int(request.form['year'])
        predicted_el_nino = predict_el_nino(year)

    return render_template('el_nino.html', predicted_el_nino=predicted_el_nino)

@app.route('/landslide', methods=['GET', 'POST'])
def landslide():
    prediction = None
    height = "133.0"  # Default value
    slope = "18.000"  # Default value
    rainfall = "248.92"  # Default value

    if request.method == 'POST':
        height = request.form['height']
        slope = request.form['slope']
        rainfall = request.form['rainfall']

        # Call your prediction function
        prediction = predict_landslide(float(height), float(slope), float(rainfall))

    return render_template('landslide.html', prediction=prediction, height=height, slope=slope, rainfall=rainfall)

def predict_landslide(height, slope, rainfall):
    # Create a DataFrame for the prediction
    test = pd.DataFrame({
        'height': [height],
        'slope': [slope],
        'rainfall': [rainfall]
    })

    # Make a prediction using the model
    prediction = landslide_model.predict(test)
    
    if prediction[0] == 0:
        return "No landslide risks found"
    else:
        return "Landslide risks detected"
    
@app.route('/earthquake')
def earthquake():
    return render_template('earthquake.html')

@app.route('/soil')
def soil():
    return render_template('soil.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')