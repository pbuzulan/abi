# Bitcoin Trading Model Documentation

## Overview

This document outlines the usage of the **BitcoinTradingModel**, a machine learning model for predicting Bitcoin returns
using high-dimensional technical indicators.

## Model Description

The **BitcoinTradingModel** class is designed to predict Bitcoin returns using 124 technical indicators. It constructs 1000
decision trees, each trained on a random subset of features. The model includes methods for training, saving, loading,
predicting, and updating with new data.

### Training the Model

The model is trained on historical data as follows:

```python
def train(self, data, target_column='return_interval'):
# Implementation details...
```

- **data**: DataFrame containing historical data with technical indicators.
- **target_column**: Name of the column containing the target variable.

### Making Predictions

To predict returns for a specific date, use the predict method:

```python
def predict(self, data):
# Implementation details...
```

- **data**: DataFrame containing a single row with the latest data and calculated technical indicators for the previous day.

### Example Usage

1. **Predicting Next Day's Return**:
   Use the trained model to predict the return for the next day (e.g., 02/01/2023) using the latest data from
   01/01/2023.

2. **Updating the Model with New Data**:
   Once new data for 02/01/2023 is available, update the dataset with calculated indicators for this date.

3. **Predicting Future Returns**:
   Predict returns for 03/01/2023 using the updated dataset.

### Notes

- Regular updates with new data help maintain the model's accuracy.
- The model outputs trading advice (**'Long'**, **'Short'**, or **'Cash'**) based on the prediction.