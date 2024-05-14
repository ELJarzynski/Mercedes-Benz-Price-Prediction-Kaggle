import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

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
# Potok dla imputera
imputer_pipeline = make_pipeline(
    SimpleImputer(strategy='mean')
)

# Określenie kolumn dla skalera
scaler_columns = df.columns.difference(["Year Build", "Price"])

# Potok dla skalera
scaler_pipeline = make_pipeline(
    MinMaxScaler()
)

# Przetworzenie danych za pomocą potoków
processed_data = imputer_pipeline.fit_transform(df)

# Utworzenie DataFrame z przetworzonych danych
df = pd.DataFrame(processed_data, columns=df.columns)

# Przetworzenie danych za pomocą potoku dla skalera
df[scaler_columns] = scaler_pipeline.fit_transform(df[scaler_columns])
print(df.head())
print(df.info())
""" ---------------------------------------- SECOND PART ---------------------------------------- """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

"""Train test split został zastosowany do podziału zbioru danych na zbiór treningowy i testowy"""
# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price']), df['Price'],
                                                    test_size=0.2, random_state=42)

"""Podział zbioru treningowego na treningowy i walidacyjny"""
# Zastosowałem walidacje krzyżową, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych
# na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

"""Inicjalizacja modelu"""
# Inicjalizacja i dopasowanie modelu regresji liniowej
lm = LinearRegression(n_jobs=-1, )
lm.fit(X_train, y_train)

# Przewidywanie na zbiorze walidacyjnym
y_pred_val = lm.predict(X_val)

# Obliczenie błędów na zbiorze walidacyjnym
mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)

# Wydruk informacji o błędach na zbiorze walidacyjnym
print(f'Linear Regression means errors\n MSE: {mse}, MAE: {mae}')
