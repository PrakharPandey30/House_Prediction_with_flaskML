from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import datetime

app = Flask(__name__, template_folder='templates')
data = pd.read_csv('7&8_data.csv')
model = pickle.load(open('model_rf.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bdrm = request.form.get('bedrooms')
    btrm = request.form.get('bathrooms')
    sf_liv = request.form.get('sqft_living')
    sf_lot = request.form.get('sqft_lot')
    flr = request.form.get('floors')
    water = request.form.get('waterfront')
    view = int(request.form.get('view', 0))
    condition = int(request.form.get('condition', 0))  # Use 0 as the default value if 'condition' is missing
    sf_ab = request.form.get('sqft_above')
    sf_bs = request.form.get('sqft_basement')
    year = datetime.datetime.now().year
    city = int(request.form.get('city', 0))
    zip = int(request.form.get('zipcode', 0))
    country = int(request.form.get('country', 0))
    build = int(request.form.get('yr_built'))
    reno = int(request.form.get('yr_renovated'))

    features = np.array([year, bdrm, btrm, sf_liv, sf_lot, flr, water, view, condition, sf_ab, sf_bs, build, reno, city, zip, country]).reshape(1, -1)
    prediction = model.predict(features)
    price = prediction * 345.6

    # Display the predicted result in an alert and redirect to the home page
    alert_message = f"Predicted Price is: {np.round(price, 2)}"
    return f"<script>alert('{alert_message}'); window.location.href='/';</script>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090)
