# Prediction of Used Mercedes-Benz Prices in America Using Machine Learning

In this project, I aim to predict the prices of used Mercedes-Benz vehicles sold in America with minimal error using a dataset from 
[Kaggle](https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset/data), 
utilizing the Scikit-learn library to train the machine learning model.

## Preparing the Dataset for Machine Learning

### Installing Required Libraries
```bash
pip install pandas
pip install sklearn
pip install numpy
```
Load the file using pandas read_csv:
```python
import pandas as pd
df = pd.read_csv('https://github.com/ELJarzynski/FinalProject-UM/blob/master/usa_mercedes_benz_prices.csv')
```

## Data Preparation for Machine Learning
### After loading the file, I used the Pandas library to clean the dataset from unnecessary characters and strings:

```python
df['Mileage'] = df['Mileage'].str.replace('mi.', '')
df['Mileage'] = df['Mileage'].str.replace(',', '')
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = df['Price'].str.replace(',', '')
df['Review Count'] = df['Review Count'].str.replace(',', '')
```
### I added a new column 'Year Build' for easier future price prediction and removed the 'Name' column, as it contained object names that are not relevant for data analysis:
```python
df['Year Build'] = df['Name'].str.split().str[0]
df = df.drop(columns=['Name'])
```
### Replaced 'Not Priced' values with None to handle missing data more easily during further processing:
```python
df.replace('Not Priced', None, inplace=True)
```
### Converted columns to float type to ensure uniform data types:
```python
df['Mileage'] = df['Mileage'].astype(float)
df['Price'] = df['Price'].astype(float)
df['Review Count'] = df['Review Count'].astype(float)
df['Year Build'] = df['Year Build'].astype(float)
```
# Using Scikit-learn for Data Standardization and Normalization with Pipelines
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler()
)

# Process data with the pipeline
processed_data = pipeline.fit_transform(df)

# Create DataFrame from processed data
df = pd.DataFrame(processed_data, columns=df.columns)
```

## The dataset looks as follows:
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/DataFrame.png)

## Using Scikit-learn to Split Data into Training, Validation, and Test Sets
### Cross-validation was used to minimize the impact of random data splits on model quality, and the results are averaged to provide a more stable estimate of model performance:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price']), df['Price'],
                                                    test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
```
# The data visualizations are as follows:
## Histograms for All Columns
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Mileage.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Price.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Rating.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Review%20Count.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Year%20Build.png)
## Scatter Plots for Price Dependency on Various Columns
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Ocena.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Oglądanie.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Rok.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Przebieg.png)
# For model initialization, I chose regression since the problem involves predicting a numerical value (price).
## The first model chosen was Linear Regression
```python
from sklearn.linear_model import LinearRegression


lm = LinearRegression(n_jobs=-1, fit_intercept=True)
lm.fit(X_train, y_train)

y_pred_val = lm.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
```
### MSE and MAE errors are:
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/LMpred.png)


### he second regression model chosen was K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=41,  # Według mnie idealna wartość gdzie jednostka jest wytrenowana na granicy przetrenowania
    weights='distance',
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
```
### MSE and MAE errors are:
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/KNNpred.png)
### The third regression model chosen was Decision Tree Regressor
```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(
    criterion='friedman_mse',
    splitter='best',
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.3,
    random_state=42,
    max_leaf_nodes=50,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
```
### MSE and MAE errors are:
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Treepred.png)
### The fourth regression model chosen was Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=90,
    criterion='friedman_mse',
    max_depth=35,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='log2',
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
```
### MSE and MAE errors are:
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/RFpred.png)

# The best regression model for predicting Mercedes-Benz prices in America is RandomForestRegressor
### With Huber loss error of 19466.369, I chose this metric because the dataset contains outliers, and the Huber loss function is more robust to outliers compared to MSE.
