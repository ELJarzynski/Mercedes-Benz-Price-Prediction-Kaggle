# Predykcja cen mieszkań mercedesów za pomcą uczenia maszynowego
W moim projekcie będę predyktował ceny Mercedesów sprzedanych w Amerycy za pomocą zbioru danych z [Kaggle](https://www.kaggle.com/datasets/danishammar/usa-mercedes-benz-prices-dataset/data) 
używając biblioteki Scikit-learn do nauczenia maszynowego tego modelu
## Przygotowanie bazy danych do wdrożenia uczenia maszynowego
### Instalacja potrzebnych bibliotek
```bash
pip install pandas
pip install sklearn

```~~~~
Wczytujemy plik za pomocą pandas read.csv 
```python
house = pd.read_csv('https://github.com/ELJarzynski/FinalProject-UM/blob/master/usa_mercedes_benz_prices.csv')

```
### Używanie sklearn do standaryzacji i normalizacji danych
Po wczytaniu pliku użyłem OneHotEncoder do przekształcenia zmiennych kategorycznych przed dalszą analizą danych
```python
from sklearn.preprocessing import OneHotEncoder
hot_encoder = OneHotEncoder()
encoded_feature = hot_encoder.fit_transform(house[['Neighborhood']])
onehot_df = pd.DataFrame(
    encoded_feature.toarray(),
    columns=hot_encoder.get_feature_names_out(['Neighborhood']),
    index=house.index
)
house.drop(['Neighborhood'], axis=1, inplace=True)
house = house.join(onehot_df)
```
Po przekształcenia zmiennych kategorycznych, za pomocą StandardScaler normalizuje kolumne SquareFeet i ustawiam dane w kolejności hronologicznej
```python
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()
SquareFeet_scaler = StandardScaler.fit_transform(house[['SquareFeet']])
SquareFeet_scaler_df = pd.DataFrame(SquareFeet_scaler, columns=['SquareFeet'])
house['SquareFeet'] = SquareFeet_scaler_df['SquareFeet']

df = house.sort_values(by='YearBuilt')
df = df.reset_index(drop=True)
```
![alt table](https://github.com/ELJarzynski/Inzynirka/blob/main/images/Terminal%20after%20data%20prepering.jpg)
## Gdy już dane są przygotwane używam framework'a pytorch do uczeniam maszynowego
### Za pomocą sklearn dzielimy dane na treningowe i testowe
```python
from sklearn.model_selection import train_test_split
X = df.drop(['Price'], axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
# This will be continued :)
