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

pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler()
)

# Przetworzenie danych za pomocą potoków
processed_data = pipeline.fit_transform(df)

# Utworzenie DataFrame z przetworzonych danych
df = pd.DataFrame(processed_data, columns=df.columns)
```

## Zbiór danych prezentuje się następująco 
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/DataFrame.png)

## Za pomocą sklearn dzielimy dane na zbiór treningowy, walidacyjny i testowy
### Została zastosowania walidacja krzyżowa, ponieważ pozwala ona na zminimalizowanie wpływu losowego podziału danych na jakość modelu oraz wyniki są uśredniane, co daje bardziej stabilną ocenę jakości modelu
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['Price']), df['Price'],
                                                    test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
```
# Tak prezentują się wykresy danych
## Histogramy dla wszystkich kolumn
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Mileage.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Price.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Rating.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Review%20Count.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Year%20Build.png)
## Wykres scatter dla zależności ceny od poszczególnych kolumn
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Ocena.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Oglądanie.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Rok.png)
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Cena%20Przebieg.png)
# Na inicjalizacje Modeli wybrałem Regresje, ponieważ problem polega na przewidywaniu ceny, która jest zmienną ciągłą
## Na pierwszy model wybrałem regresje liniową
```python
lm = LinearRegression(n_jobs=-1, fit_intercept=True)
lm.fit(X_train, y_train)

y_pred_val = lm.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
mae = mean_absolute_error(y_val, y_pred_val)
```
### Błąd MSE i MAE jest równy
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/LMPred.png)


### Drugi rodzaj regresji KNN
```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(
    n_neighbors=41,  # Według mnie idealna wartość gdzie jednostka jest wytrenowana na granicy przetrenowania
    weights='distance',
    n_jobs=-1  # liczba równoległych wątków
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
```
### Błąd MSE i MAE jest  jest równy
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/KNNpred.png)
### Trzeci rodzaj regresji DecisionTreeRegressor
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
### Błąd MSE i MAE jest  jest równy
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/Treepred.png)
### Czwarty rodzaj regresji RandomForestRegressor
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
### Błąd MSE i MAE jest  jest równy
![alt table](https://github.com/ELJarzynski/FinalProject-UM/blob/master/photos/RFpred.png)

# Najlepszą regresją do predykcji cen Mercedesów w Ameryce jest RandomForestRegressor z wynikiem 
## Średnio kwadratowym wynoszącym 0.019 i błędem absolutnym wynoszącym 0.088
