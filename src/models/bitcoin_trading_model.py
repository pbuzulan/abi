import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# based on Predicting bitcoin returns using high-dimensional technical indicators research
class BitcoinTradingModel:
    def __init__(self, num_trees=1000, num_features=11):
        self.num_trees = num_trees
        self.num_features = num_features
        self.tree_models = []
        self.threshold = None

    def save(self, filename):
        """
        Save the trained model to a file.

        Parameters:
            filename (str): File path to save the model.

        Returns:
            None
        """
        model_data = {
            'num_trees': self.num_trees,
            'num_features': self.num_features,
            'tree_models': self.tree_models,
            'threshold': self.threshold
        }
        joblib.dump(model_data, filename)

    @classmethod
    def load(cls, filename):
        """
        Load a trained model from a file.

        Parameters:
            filename (str): File path to load the model from.

        Returns:
            BitcoinTradingModel: A trained model instance.
        """
        model_data = joblib.load(filename)
        loaded_model = cls(
            num_trees=model_data['num_trees'],
            num_features=model_data['num_features']
        )
        loaded_model.tree_models = model_data['tree_models']
        loaded_model.threshold = model_data['threshold']
        return loaded_model

    def train(self, data):
        # Assuming 'data' is a DataFrame containing historical Bitcoin data with technical indicators

        # Define the features and target variable
        X = data.drop(columns=['return'])  # Replace 'return' with the actual column name
        y = data['return']  # Replace 'return' with the actual column name

        # Train 1000 decision trees
        for _ in range(self.num_trees):
            # Randomly select a subset of features for each tree
            selected_features = np.random.choice(X.columns, self.num_features, replace=False)
            X_subset = X[selected_features]

            # Split the data into training and test sets (adjust the split ratio as needed)
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            # Create and train a DecisionTreeClassifier
            tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
            tree.fit(X_train, y_train)

            # Append the trained tree to the list
            self.tree_models.append(tree)

        # Calculate the average threshold based on the trained trees
        self.calculate_threshold(data)

    def calculate_threshold(self, data):
        # Calculate the average prediction from the 1000 trees
        average_predictions = np.mean([tree.predict(data.drop(columns=['return'])) for tree in self.tree_models],
                                      axis=0)

        # Calculate the threshold based on the yb values for which signals are generated
        self.threshold = np.mean([abs(yb) for yb in average_predictions])

    def predict(self, data):
        """
        Predict Bitcoin trading signals.

        Parameters:
            data (DataFrame): DataFrame containing the latest Bitcoin data for prediction.

        Returns:
            np.array: Predicted trading signals (1 for buy, 0 for hold/sell).
        """
        # Calculate the average prediction from the trained trees
        average_predictions = np.mean([tree.predict(data) for tree in self.tree_models], axis=0)

        # Generate trading signals based on the threshold
        trading_signals = np.where(average_predictions > self.threshold, 1, 0)

        return trading_signals

    def update(self, X_new, y_new):
        """
        Update the existing model with new data.

        Parameters:
            X_new (DataFrame): New data features.
            y_new (Series): New data target values.

        Returns:
            None
        """
        if not self.tree_models:
            # If the model is not yet trained, train it with the new data
            self.train(pd.concat([X_new, y_new], axis=1))
        else:
            # Otherwise, update the model with the new data incrementally
            for tree in self.tree_models:
                tree.partial_fit(X_new, y_new, classes=[0, 1])
            # Recalculate the threshold based on the updated model
            self.calculate_threshold(pd.concat([X_new, y_new], axis=1))
