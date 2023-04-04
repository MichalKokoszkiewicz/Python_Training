def silnia(liczba):
    if liczba == 0:
        return 1
    return liczba * silnia(liczba - 1)

#silnia(3) -> 3 * silnia(2)
#silnia(2) -> 2 * slinia(1)
#silnia(1) -> 1 * silnia(0)
#silnia(0) -> 1
#silnia(1) -> 1 * 1
#silnia(2) -> 2 * 1
#silnia(3) -> 3 * 2
# 6

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

# i = 0
# while True:
#     print(str(i) + "! = " + str(silnia(i)))
#     i += 1
