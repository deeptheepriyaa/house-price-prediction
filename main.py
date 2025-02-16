import joblib
import pandas as pd

# Load trained model
model = joblib.load("models/house_price_model.pkl")

# Example new data (OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt)
new_house = pd.DataFrame([[7, 2000, 2, 1000, 2, 2005]], 
                         columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"])

# Make prediction
predicted_price = model.predict(new_house)
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
