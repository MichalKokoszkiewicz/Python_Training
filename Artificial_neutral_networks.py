import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from sklearn import model_selection 
import matplotlib.pyplot as plt

# Wczytujemy plik iris z danymi za pomocą biblioteki pandas 
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


data = iris.values # Tworzymy tablice za pomocą biblioteki numpy i atrybutu values, która zawiera dane z pliku iris 
# Tworzymy 2 tablice  
X = data[:,:-1] # Tablica X zawiera dane wejściowe do naszej sztucznej sieci neuronowej, czyli 4 pierwsze kolumny z pliku iris 
y = data[:,-1] # Tablica y zawiera wartości wzorcowe, czyli ostatnią kolumnę z pliku iris

# Tworzymy etykiety dla naszych wartości wzorcowych z tablicy y za pomocą biblioteki sklearn i funkcji OrdinalEncoder
# Dla Iris-setosa nadaje etykiete 0, dla Iris-versicolor nadaje etykiete 1, dla Iris-virginica nadaje etykiete 2
y_iris = OrdinalEncoder() # Tworzymy tablice OrdinalEncoder
y_etykieta=y_iris.fit_transform(y.reshape(150,1)) # Korzystamy z metody fit_transform, która dopasowuje do danych, a następnie je przekształca i tworzymy tablicę z etykietami 

# Za pomocą biblioteki sklearn i metody train_test_split dzielimy dane na 2 zestawy: uczący się(2/3) i testowy(1/3), używamy random_state do losowości wyboru danych do zestawów
splits = model_selection.train_test_split(X, y_etykieta, test_size=.333, random_state=0) 
X_train, X_test, y_train, y_test = splits

# Zmieniamy typ danych na float32, żeby biblioteka tensorflow prawidłowo działała
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

                                        #***            Sieć 1          ***#

siec1=tf.keras.models.Sequential() # Tworzymy model senkwencyjny za pomocą biblioteki tensorflow
siec1.add(tf.keras.layers.Input(shape=(4))) # Tworzymy pierwszą warstwę, która odczytuje dane, wybieramy ile ma być neuronów wejściowych
# Kolejne warstwy, zawierają ilość neuronów i funkcję aktywacji 
siec1.add(tf.keras.layers.Dense(3,'tanh')) 
siec1.add(tf.keras.layers.Dense(4,'tanh'))
siec1.add(tf.keras.layers.Dense(3,'tanh'))
siec1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy()) # Kompilacja danych
siec1.summary() # Podsumowanie danych do analizy i wizualizacji 
siec1.get_weights() # Zwraca bieżące wagi warstwy jako tablice


model1=siec1.fit(X_train,y_train.flatten(),epochs=7) # Uczenie się sieci, określenie ilości epochs(ilość iteracji)
y_out=siec1.predict(X_test) # Odczytanie wyjścia 
y_out_etykieta=np.argmax(y_out, axis=1)
# Sprawdzamy czy etykieta oczekiwana jest taka sama jak eytkieta ze zbioru testowego, jeżeli jest taka sama to jest 1 i dzielenie daje nam % prawidłowych odpowiedzi
print(sum(y_out_etykieta==y_test.flatten()))
sum(y_out_etykieta==y_test.flatten())/y_test.shape[0] 

# Wykres, który pokazuje skuteczność naszej sieci 
model1.history.items()
plt.plot(model1.history['loss'])

# Wykres wartości oczekiwanych i wychodzących
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y_out_etykieta,'o')
ax.plot(y_test.flatten(),'*')

                                        #****           Sieć 2          ***#

siec2=tf.keras.models.Sequential() # Tworzymy 2 model senkwencyjny za pomocą biblioteki tensorflow
siec2.add(tf.keras.layers.Input(shape=(4))) # Tworzymy pierwszą warstwę, która odczytuje dane, wybieramy ile ma być neuronów wejściowych
# Kolejne warstwy, zawierają ilość neuronów i funkcję aktywacji 
siec2.add(tf.keras.layers.Dense(4,'sigmoid'))
siec2.add(tf.keras.layers.Dense(3,'sigmoid')) 
siec2.add(tf.keras.layers.Dense(4,'sigmoid'))
siec2.add(tf.keras.layers.Dense(2,'sigmoid'))
siec2.add(tf.keras.layers.Dense(3,'sigmoid'))
siec2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy()) # Kompilacja danych
siec2.summary() # Podsumowanie danych do analizy i wizualizacji 
siec2.get_weights() # Zwraca bieżące wagi warstwy jako tablice


model2=siec2.fit(X_train,y_train.flatten(),epochs=10) # Uczenie się sieci, określenie ilości epochs(ilość iteracji)
y_out=siec2.predict(X_test) # Odczytanie wyjścia 
y_out_etykieta=np.argmax(y_out, axis=1)
# Sprawdzamy czy etykieta oczekiwana jest taka sama jak eytkieta ze zbioru testowego, jeżeli jest taka sama to jest 1 i dzielenie daje nam % prawidłowych odpowiedzi
print(sum(y_out_etykieta==y_test.flatten()))
sum(y_out_etykieta==y_test.flatten())/y_test.shape[0] 


# Wykres, który pokazuje skuteczność naszej sieci 
model2.history.items()
plt.plot(model2.history['loss'])

# Wykres wartości oczekiwanych i wychodzących
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y_out_etykieta,'o')
ax.plot(y_test.flatten(),'*')

                                        #***        Sieć 3          ***#

siec3=tf.keras.models.Sequential() # Tworzymy 3 model senkwencyjny za pomocą biblioteki tensorflow
siec3.add(tf.keras.layers.Input(shape=(4))) # Tworzymy pierwszą warstwę, która odczytuje dane, wybieramy ile ma być neuronów wejściowych
# Kolejne warstwy, zawierają ilość neuronów i funkcję aktywacji 
siec3.add(tf.keras.layers.Dense(4,'softsign'))
siec3.add(tf.keras.layers.Dense(3,'softsign'))
siec3.add(tf.keras.layers.Dense(3,'softsign')) 
siec3.add(tf.keras.layers.Dense(4,'softsign'))
siec3.add(tf.keras.layers.Dense(3,'softsign'))
siec3.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy()) # Kompilacja danych
siec3.summary() # Podsumowanie danych do analizy i wizualizacji 
siec3.get_weights() # Zwraca bieżące wagi warstwy jako tablice


model3=siec3.fit(X_train,y_train.flatten(),epochs=12) # Uczenie się sieci, określenie ilości epochs(ilość iteracji)
y_out=siec3.predict(X_test) # Odczytanie wyjścia 
y_out_etykieta=np.argmax(y_out, axis=1)
# Sprawdzamy czy etykieta oczekiwana jest taka sama jak eytkieta ze zbioru testowego, jeżeli jest taka sama to jest 1 i dzielenie daje nam % prawidłowych odpowiedzi
print(sum(y_out_etykieta==y_test.flatten()))
sum(y_out_etykieta==y_test.flatten())/y_test.shape[0] 


# Wykres, który pokazuje skuteczność naszej sieci 
model3.history.items()
plt.plot(model3.history['loss'])

# Wykres wartości oczekiwanych i wychodzących
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y_out_etykieta,'o')
ax.plot(y_test.flatten(),'*') 


                                        #***        Sieć 4          ***#

siec4=tf.keras.models.Sequential() # Tworzymy 4 model senkwencyjny za pomocą biblioteki tensorflow
siec4.add(tf.keras.layers.Input(shape=(4))) # Tworzymy pierwszą warstwę, która odczytuje dane, wybieramy ile ma być neuronów wejściowych
# Kolejne warstwy, zawierają ilość neuronów i funkcję aktywacji 
siec4.add(tf.keras.layers.Dense(4,'softplus'))
siec4.add(tf.keras.layers.Dense(3,'softplus'))
siec4.add(tf.keras.layers.Dense(3,'softplus')) 
siec4.add(tf.keras.layers.Dense(4,'softplus'))
siec4.add(tf.keras.layers.Dense(3,'softplus'))
siec4.add(tf.keras.layers.Dense(4,'softplus'))
siec4.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy()) # Kompilacja danych
siec4.summary() # Podsumowanie danych do analizy i wizualizacji 
siec4.get_weights() # Zwraca bieżące wagi warstwy jako tablice


model4=siec4.fit(X_train,y_train.flatten(),epochs=20) # Uczenie się sieci, określenie ilości epochs(ilość iteracji)
y_out=siec4.predict(X_test) # Odczytanie wyjścia 
y_out_etykieta=np.argmax(y_out, axis=1)
# Sprawdzamy czy etykieta oczekiwana jest taka sama jak eytkieta ze zbioru testowego, jeżeli jest taka sama to jest 1 i dzielenie daje nam % prawidłowych odpowiedzi
print(sum(y_out_etykieta==y_test.flatten()))
sum(y_out_etykieta==y_test.flatten())/y_test.shape[0] 


# Wykres, który pokazuje skuteczność naszej sieci 
model4.history.items()
plt.plot(model4.history['loss'])

# Wykres wartości oczekiwanych i wychodzących
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(y_out_etykieta,'o')
ax.plot(y_test.flatten(),'*') 