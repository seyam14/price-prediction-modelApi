import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate a large dataset with 10,000 samples
n_samples = 10000

# Generate sqft between 500 and 4000
sqft = np.random.randint(500, 4000, size=n_samples)

# Generate bedrooms between 1 and 6
bedrooms = np.random.randint(1, 7, size=n_samples)

# Simulate price with some randomness but generally correlated to sqft and bedrooms
# Base price per sqft and per bedroom coefficients
price = (sqft * 150) + (bedrooms * 10000) + np.random.normal(0, 20000, size=n_samples)

# Create DataFrame
df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'price': price
})

# Features and target
X = df[['sqft', 'bedrooms']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")

# Optional: show first few rows
print(df.head())
