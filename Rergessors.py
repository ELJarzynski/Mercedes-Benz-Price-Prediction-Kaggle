from sklearn.linear_model import LinearRegression
from datasetPreparing import *
from sklearn.neighbors import KNeighborsRegressor

"""Inicjalizacja modelu"""

model = LinearRegression(n_jobs=-1, fit_intercept=True)
model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)

print(f'Linear Regression means errors\n MSE: {mse}, MAE: {mae}')

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

print(f'KNN means errors\nMSE: {mse}, MAE: {mae}')

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
print(f'DecisionTreeRegressor means errors\nMSE: {mse}, MAE: {mae}')