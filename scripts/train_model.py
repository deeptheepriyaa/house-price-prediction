import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_preprocessing import load_data  # Import the data preprocessing function

# Load preprocessed data
X_train, X_test, y_train, y_test = load_data()

# Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Train the model using training data

# Predict house prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Save the trained model to the models/ directory
joblib.dump(model, "models/house_price_model.pkl")
print("Model saved successfully as 'models/random_forest.pkl'")
