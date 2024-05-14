import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

""" ---------------------------------------- DATA PREPARING ---------------------------------------- """

"""Settings of terminal setup"""
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

file_directory = r"C:\Users\kamil\Desktop\Studia\Semestr VI\MachinLearning\Kamil-Jarzynski-164395-USA-Mercedes-Benz-Prices\usa_mercedes_benz_prices.csv"
df = pd.read_csv(file_directory)

"""Removing redundant characters and strings from data"""
# Usuwam zbędne znaki oraz ciągi znaków z danych, aby umożliwić konwersję wszystkich kolumn na typ float.
df['Mileage'] = df['Mileage'].str.replace('mi.', '')
df['Mileage'] = df['Mileage'].str.replace(',', '')
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = df['Price'].str.replace(',', '')
df['Review Count'] = df['Review Count'].str.replace(',', '')

"""Adding new column filled by year of build"""
# Tworzę nową kolumnę "Year Build", która zawiera informacje o roku produkcji Mercedesa, co może być przydatne do przewidywania cen.
df['Year Build'] = df['Name'].str.split().str[0]
df['Name'] = df['Name'].str.split(n=1).str[1]

"""Converting price value 'Not Priced' to None"""
# Zmieniam wartości 'Not Priced' na None, aby móc łatwiej obsłużyć brakjące dane w dalszej obróbce.
df.replace('Not Priced', None, inplace=True)

"""Removing column Name"""
# Usuwam kolumne "Name", ponieważ nie jest ona potrzebna do przewidywania cen pojazdów.
df = df.drop(columns=['Name'])

"""Converting columns to float values"""
# Konwetuje kolumny na typ float, aby miały wszystkie ten sam typ danych
df['Mileage'] = df['Mileage'].astype(float)
df['Price'] = df['Price'].astype(float)
df['Review Count'] = df['Review Count'].astype(float)
df['Year Build'] = df['Year Build'].astype(float)

"""Data preprocessing"""
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler()
)

# Przetworzenie danych za pomocą potoków
processed_data = pipeline.fit_transform(df)

# Utworzenie DataFrame z przetworzonych danych
df = pd.DataFrame(processed_data, columns=df.columns)

""" ---------------------------------------- SECOND PART ---------------------------------------- """
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""Train test split was used to split dataset into training and testing sets"""
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price']), df['Price'],
                                                    test_size=0.33, random_state=42)

"""Splitting the training set into training and validation sets"""
# Zastosowałem walidacje krzyżową, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych
# na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)

"""Inicjalizacja modelu"""
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

"""Predictions errors"""
import numpy as np


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    huber_loss = np.where(np.abs(error) < delta, 0.5 * error ** 2, delta * (np.abs(error) - 0.5 * delta))
    return np.mean(huber_loss)

huber_val = huber_loss(y_val, y_pred)
print(f'RandomForestRegressor means errors\nMSE: {mse:.3f}, MAE: {mae:.3f}, Huber Prediction: {huber_val:.3f}')
