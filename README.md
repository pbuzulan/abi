# Auto Bitinvest Investments

### Pre requisites

```
brew install ta-lib
```

### Setup python environment

```
 python3 -m venv venv
 . venv/bin/activate
 pip3 install --upgrade pip
 pip3 install -r requirements.txt
```

### Setup python environment in PyCharm

```
Preferences -> Project -> Project Interpreter -> Add -> Existing environment -> ~~~~.../venv/bin/python3
```

### Prepare raw dataset

```
Download raw dataset from S3 and save it to `data/raw` folder.
The file name should be BTC_ohlc_day.csv
```

### Run BitcoinTradingModel

#### Training the model

```
run `src/train_model.py` file
```

#### Predicting the model

```
run `src/predict_model.py` file
```

#### Simulate return

```
run `src/simulate_returns.py` file
```


