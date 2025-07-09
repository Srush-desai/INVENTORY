import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump

# ---- Customer Segmentation ----
cust_df = pd.read_csv('datasets/customers.csv')
cust_X = cust_df[['Annual Income (k$)', 'Spending Score (1-100)']]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(cust_X)
dump(kmeans, 'models/cluster_model.pkl')

# ---- Inventory Distribution ----
store_df = pd.read_csv('datasets/store_data.csv')
store_df['Date'] = pd.to_datetime(store_df['Date'])
store_df['Month'] = store_df['Date'].dt.month

grouped = store_df.groupby(['Store ID', 'Product ID', 'Month'])['Units Sold'].sum().reset_index()

X = grouped[['Store ID', 'Product ID', 'Month']].copy()
X['Store ID'] = X['Store ID'].str.extract(r'(\d+)').astype(int)
X['Product ID'] = X['Product ID'].str.extract(r'(\d+)').astype(int)
y = grouped['Units Sold']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
dump(rf, 'models/stock_model.pkl')

# ---- Demand Forecasting ----
def train_demand_model():
    df = pd.read_csv('datasets/sales_data_sample.csv', encoding='ISO-8859-1')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Product ID'] = df['Product ID'].astype(str).str.extract(r'(\d+)').astype(int)
    df['Store ID'] = df['Store ID'].astype(str).str.extract(r'(\d+)').astype(int)

    X = df[['Price', 'Promotion', 'Month']]
    y = df['Units Sold']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    dump(model, 'models/demand_model.pkl')
    print("✅ demand_model.pkl saved successfully.")

# Run demand model training only when script is executed directly

