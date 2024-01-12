import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def train_model(df, target_column='close', save_model=False):
    """
    Train an XGBRegressor model on the given DataFrame and save the trained model.

    Parameters:
    df (DataFrame): The DataFrame containing the features and target.
    target_column (str): The name of the target column.
    model_filename (str): The filename to save the trained model.

    Returns:
    dict: A dictionary containing RMSE, MSE, MAE, and R^2 metrics.
    """
    # Split the data
    X = df.drop([target_column, 'datetime_utc', 'timestamp'], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the model
    model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01, n_estimators=100)
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Save the model
    if save_model:
        save_trained_model(model)

    # Return the evaluation metrics
    return {
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mae,
        'R^2': r2
    }


def save_trained_model(model, filename='crypto_price_prediction_model.pkl'):
    """
    Saves the trained model to a file.

    Parameters:
    model (XGBRegressor): The trained model to be saved.
    filename (str): The name of the file to save the model.
    """
    filename = '/Users/Petru_Buzulan/Private/bi/workspace/abi/models/crypto_price_prediction_model.pkl'
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")


import time

# Example usage
# Assuming 'df' is your DataFrame and you have defined a 'target' column
df = pd.read_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/training/BTC.csv')
start_time = time.time()
trained_model = train_model(df, 'close', save_model=True)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"The method took {elapsed_time} seconds to complete.\n\n")
print(trained_model)

# Usage example:
# average_preds, test_preds = train_and_predict_bitcoin_returns(bitcoin_data)
