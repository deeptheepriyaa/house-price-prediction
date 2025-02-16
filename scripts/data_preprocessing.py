import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Loads the dataset, selects key features, splits it into training and testing sets, and normalizes it.
    """
    # Load dataset from the data folder
    data = pd.read_csv("data/train.csv")
    
    # Select relevant features (important columns affecting house prices)
    features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    target = "SalePrice"

    # Split into input features (X) and target variable (y)
    X = data[features]
    y = data[target]

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize (scale) the data to improve model performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test = scaler.transform(X_test)  # Transform test data using the same scaler

    return X_train, X_test, y_train, y_test
