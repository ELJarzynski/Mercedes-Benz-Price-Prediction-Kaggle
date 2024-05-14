# Predykcja cen mieszkań mercedesów za pomcą uczenia maszynowego
W moim projekcie będę predyktował ceny Mercedesów sprzedanych w Amerycy za pomocą zbioru danych z 
[Kaggle](https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset/data) 
używając biblioteki Scikit-learn do nauczenia maszynowego tego modelu
## Przygotowanie bazy danych do wdrożenia uczenia maszynowego
### Instalacja potrzebnych bibliotek
```bash
pip install pandas
pip install sklearn

```
Wczytujemy plik za pomocą pandas read.csv 
```python
import pandas as pd
df = pd.read_csv('https://github.com/ELJarzynski/FinalProject-UM/blob/master/usa_mercedes_benz_prices.csv')
```

## Przygotowanie danych pod uczenie maszynowe
### Po wczytaniu pliku użyłem biblioteki Pandas do wyczyszczenia zbióru danych ze zbędnych znaków i stringów poczym dodałem nową kolumne 'Year Build' i usunąłem kolumne 'Name', ponieważ zawierała nazwy obiektów, które nie są istotne dla analizy danych.

```python
df['Mileage'] = df['Mileage'].str.replace('mi.', '')
df['Mileage'] = df['Mileage'].str.replace(',', '')
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = df['Price'].str.replace(',', '')
df['Review Count'] = df['Review Count'].str.replace(',', '')
```
```python
df['Year Build'] = df['Name'].str.split().str[0]
df['Name'] = df['Name'].str.split(n=1).str[1]
df = df.drop(columns=['Name'])
```
### Zmiana wartości 'Not Priced' na None, aby móc łatwiej obsłużyć brakjące dane w dalszej obróbce.
```python
df.replace('Not Priced', None, inplace=True)
```
### Konwersja kolumn na typ float, aby miały wszystkie ten sam typ danych
```python
df['Mileage'] = df['Mileage'].astype(float)
df['Price'] = df['Price'].astype(float)
df['Review Count'] = df['Review Count'].astype(float)
df['Year Build'] = df['Year Build'].astype(float)
```
# Używanie sklearn do standaryzacji i normalizacji danych przy użyciu potoków
```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

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
```

## Zbiór danych prezentuje się następująco 
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/DataFrame.png)

## Za pomocą sklearn dzielimy dane na zbiór treningowy, walidacyjny i testowy
### Została zastosowania walidacja krzyżowa, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price']), df['Price'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
```
# Tak prezentują się wykresy danych
## Histogramy dla wszystkich kolumn
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Mileage.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Price.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Rating.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Review%20Count.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Year%20Build.png)
## Wykres scatter dla zależności ceny od poszczególnych kolumn
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Price.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Rating.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Review%20Count.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Year%20Build.png)
# Na tym inicjalizacji Modeli wybrałem Regresje, ponieważ problem polega na przewidywaniu ceny, która jest zmienną ciągłą
## Na pierwszy model wybrałem regresje liniową bo prędzej dużo z nią pracowałem
```python
# Inicjalizacja i dopasowanie modelu regresji liniowej
lm = LinearRegression(n_jobs=-1, fit_intercept=True)
lm.fit(X_train, y_train)

# Predykcja na zbiorze walidacyjnym
y_pred_val = lm.predict(X_val)

# Obliczenie błędów na zbiorze walidacyjnym
mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
```
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/LinearRegressionPred.png)
### Błąd MSE i MAE jest dość wysoki
