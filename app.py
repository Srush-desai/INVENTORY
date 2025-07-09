from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# --- Customer Segmentation ---
@app.route('/segment', methods=['GET', 'POST'])
def segment():
    df = pd.read_csv('datasets/customers.csv')
    model = load('models/cluster_model.pkl')
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    df['Cluster'] = model.predict(X)
    return render_template('segment.html', tables=[df.head().to_html(classes='data', header=True, index=False)])

# --- Demand Forecasting ---
@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    prediction = None
    if request.method == 'POST':
        price = float(request.form['price'])
        promotion = int(request.form['promotion'])
        month = int(request.form['month'])

        model = load('models/demand_model.pkl')
        pred = model.predict([[price, promotion, month]])
        prediction = round(pred[0], 2)
        

    return render_template('forecast.html', prediction=prediction)

# --- Inventory Distribution ---
@app.route('/distribute', methods=['GET', 'POST'])
def distribute():
    df = pd.read_csv('datasets/store_data.csv')

    # Convert Date to datetime and extract Month
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month

    # Clean Store ID and Product ID (keep column names unchanged!)
    df['Store ID'] = df['Store ID'].str.extract(r'(\d+)').astype(int)
    df['Product ID'] = df['Product ID'].str.extract(r'(\d+)').astype(int)

    # Group by original column names
    grouped = df.groupby(['Store ID', 'Product ID', 'Month'])['Units Sold'].sum().reset_index()

    # Use the same column names the model was trained on
    X = grouped[['Store ID', 'Product ID', 'Month']]

    # Load model and predict
    model = load('models/stock_model.pkl')
    y_pred = model.predict(X)
    grouped['Predicted Units'] = y_pred

    return render_template('distribute.html', tables=[grouped.to_html(classes='data', header=True, index=False)])

if __name__ == '__main__':
    app.run(debug=True)
