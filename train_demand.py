import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

# Sample training data — replace with your actual dataset later
data = {
    'price': [100, 120, 130, 150, 170, 180],
    'promotion': [0, 1, 0, 1, 0, 1],
    'month': [1, 2, 3, 4, 5, 6],
    'demand': [220, 250, 230, 260, 240, 270]
}

df = pd.DataFrame(data)

X = df[['price', 'promotion', 'month']]
y = df['demand']

model = LinearRegression()
model.fit(X, y)

# Save the model
dump(model, 'models/demand_model.pkl')
print("✅ Model saved successfully to 'models/demand_model.pkl'")
