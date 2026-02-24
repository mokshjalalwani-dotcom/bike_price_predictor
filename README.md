#  Used Bike Resale Price Predictor
A sleek Streamlit app that predicts resale prices of used bikes using an XGBoost model and preprocessing pipeline.

## Features
- Choose brand, model series, city, kilometers driven, bike age, power, and owner type.
- Get a quick predicted resale price.
- Interactive feature importance chart to understand what factors influence predictions.

## Files
- `app.py` — Streamlit app
- `Used_Bikes.csv` — dataset used for dropdowns / preprocessing
- `bike_price_model.pkl` — trained ML model
- `requirements.txt` — dependencies

##Install dependencies:

pip install -r requirements.txt

##Run the app:

streamlit run app.py

