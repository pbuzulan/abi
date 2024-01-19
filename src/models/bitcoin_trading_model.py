import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# based on Predicting bitcoin returns using high-dimensional technical indicators research
class BitcoinTradingModel:
    def __init__(self, num_trees=1000, num_features=11):
        """
        Initialize the BitcoinTradingModel.

        Parameters:
        num_trees (int): Number of decision trees to be used in the model.
        num_features (int): Number of features to be selected randomly for training each tree.

        Attributes:
        tree_models (list): Stores the trained decision tree models.
        tree_performance (list): Stores the performance of each tree model.
        return_ranges (list of tuples): Defines the 21 return ranges as per the research.
        """
        self.model_name = 'BitcoinTradingModel'
        self.model_file_name = 'ReturnIntervalPredictionBitcoinTradingModel_v1.pkl'
        self.num_trees = num_trees
        self.num_features = num_features
        self.tree_models = []
        self.tree_performance = []
        self.selected_features_per_tree = []  # Store selected features for each tree
        self.return_ranges = self._define_return_ranges()

    def _define_return_ranges(self):
        """
        Define the 21 return ranges as per the research.

        Returns:
        List of tuples representing the return ranges.
        """
        negative_ranges = [(-100, -11), (-11, -9), (-9, -7), (-7, -5), (-5, -3), (-3, -1), (-1, -0.8), (-0.8, -0.6),
                           (-0.6, -0.4), (-0.4, -0.2)]
        positive_ranges = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11),
                           (11, float('inf'))]
        neutral_range = [(-0.2, 0.2)]
        return negative_ranges + neutral_range + positive_ranges

    def save(self, filename):
        joblib.dump(self, filename)

    def update(self, new_data, target_column='return_interval'):
        X_new = new_data.drop(columns=[target_column])
        y_new = new_data[target_column]
        for tree in self.tree_models:
            tree.fit(X_new, y_new)

    @classmethod
    def load(cls, filename):
        model = joblib.load(filename)
        return model

    def train(self, data, target_column='return_interval'):
        """
        Train the model on the provided dataset.

        Parameters:
        data (DataFrame): The dataset containing features and target column.
        target_column (str): The name of the target column in the dataset.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        for _ in range(self.num_trees):
            selected_features = np.random.choice(X.columns, self.num_features, replace=False)
            self.selected_features_per_tree.append(selected_features)

            # selected_features = np.random.choice(X.columns, self.num_features, replace=True)
            X_subset = X[selected_features]
            X_train, X_val, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
            tree.fit(X_train, y_train)
            self.tree_models.append(tree)

            # Evaluate the tree's performance
            tree_performance = self.evaluate_tree(tree, X_val, y_val)
            self.tree_performance.append(tree_performance)

    def predict(self, data):
        """
        Predict the trading advice and the return range for the given data.

        Parameters:
        data (DataFrame): The dataset containing a single row with the latest data and indicators.

        Returns:
        list of tuples: Each tuple contains the trading advice ('Long', 'Short', or 'Cash') and the return range label.
        """
        # TODO: check 8 if it's ok with other datasets and tests (it's not in the research)
        reliable_long_ranges = [4, 5, 6, 7, 9, 10]  # Indices for 'Long' advice
        reliable_short_ranges = [-8, -7]  # Indices for 'Short' advice

        # Ensure data contains the same features as during training
        # and in the correct order
        predictions = []
        for tree, features in zip(self.tree_models, self.selected_features_per_tree):
            # Ensure that only the features used for this tree are in the data
            X_subset = data[features]
            predictions.append(tree.predict(X_subset))

        predictions = np.array(predictions)
        trading_advice_and_range = []

        for preds in predictions.T:
            # Aggregate predictions across all trees for the sample
            # Using mode (most common prediction) as an example
            mode_result = stats.mode(preds)

            # Check if the mode result is a scalar or an array
            if np.isscalar(mode_result.mode):
                pred = mode_result.mode
            else:
                pred = mode_result.mode[0]

            # Find the range index for the prediction
            range_label = None
            for i, (lower, upper) in enumerate(self.return_ranges):
                if lower <= pred < upper:
                    range_label = i - 10  # Adjust index to match the range labels
                    break

            # Determine trading advice based on the range label
            advice = 'Cash'  # Default advice
            if range_label in reliable_long_ranges:
                advice = 'Long'
            elif range_label in reliable_short_ranges:
                advice = 'Short'

            trading_advice_and_range.append((advice, range_label))

        return trading_advice_and_range

    def evaluate_tree(self, tree, X_val, y_val):
        """
        Evaluate the performance of a tree on the validation set.

        Parameters:
        tree (DecisionTreeClassifier): The decision tree to be evaluated.
        X_val (DataFrame): Validation set features.
        y_val (Series): Validation set target.

        Returns:
        float: The performance metric (e.g., accuracy) of the tree.
        """
        predictions = tree.predict(X_val)
        accuracy = np.mean(predictions == y_val)
        return accuracy

    # TODO: to refactor as it's a duplicate method
    def _determine_position(self, actual_range_label):
        if actual_range_label in [4, 5, 6, 7, 9, 10]:
            return 'Long'
        elif actual_range_label in [-8, -7]:
            return 'Short'
        else:
            return 'Cash'

    def calculate_metrics(self, test_data, actual_returns_interval, actual_returns):
        """
        Calculate various performance metrics based on the model's predictions.

        Args:
        test_data (DataFrame): The test dataset.
        actual_returns (Series): Actual returns corresponding to the test dataset.

        Returns:
        dict: A dictionary containing various performance metrics.
        """
        predictions = self.predict(test_data)

        # Calculate annualized volatility
        daily_volatility = np.std(actual_returns)
        # TODO: understand why annualized_volatility is so different than the research
        annualized_volatility = daily_volatility * np.sqrt(365)

        metrics = {}

        # Calculate win/loss ratios
        wins_range = losses_range = 0
        wins_position = losses_position = 0
        for pred, actual in zip(predictions, actual_returns_interval):
            predicted_position, predicted_range_label = pred

            # Find the actual range label
            actual_range_label = None
            for i, (lower, upper) in enumerate(self.return_ranges):
                if lower <= actual < upper:
                    actual_range_label = i - 10  # Adjust index to match the range labels
                    break

            # Check if the prediction matches the actual range
            if predicted_range_label == actual_range_label:
                wins_range += 1
            else:
                losses_range += 1

            # Check if the prediction matches the actual position
            if predicted_position == self._determine_position(actual_range_label):
                wins_position += 1
            else:
                losses_position += 1

        total_predictions = wins_range + losses_range
        wins_ratio_range = wins_range / total_predictions if total_predictions > 0 else 0
        losses_ratio_range = losses_range / total_predictions if total_predictions > 0 else 0
        wins_ratio_position = wins_position / total_predictions if total_predictions > 0 else 0
        losses_ratio_position = losses_position / total_predictions if total_predictions > 0 else 0
        metrics['wins_range'] = wins_range
        metrics['losses_range'] = losses_range
        metrics['wins_position'] = wins_position
        metrics['losses_position'] = losses_position
        metrics['wins_ratio_range'] = wins_ratio_range
        metrics['losses_ratio_range'] = losses_ratio_range
        metrics['wins_ratio_position'] = wins_ratio_position
        metrics['losses_ratio_position'] = losses_ratio_position
        metrics['win_loss_ration_range'] = wins_ratio_range / losses_ratio_range
        metrics['win_loss_ration_position'] = wins_ratio_position / losses_ratio_position

        return metrics
