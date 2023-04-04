def silnia(liczba):
    if liczba == 0 or liczba == 1:
        return 1
    suma = 1
    for i in range(2, liczba + 1):
        suma *= i
    return suma

print("\"x\" kończy działanie programu")
while True:
    znak = input("podaj liczbę: ")
    if znak.lower() == "x":
        break
    liczba = int(znak)
    if liczba < 0:
        print("tylko liczby dodatnie")
        continue
    print(str(liczba) + "! = " + str(silnia(liczba)))
