def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

print("\"x\" kończy działanie programu")
while True:
    znak = input("podaj liczbę: ")
    if znak.lower() == "x":
        break
    liczba = int(znak)
    if liczba < 0:
        print("tylko liczby dodatnie")
        continue
    print("fib(" + str(liczba) + ") = " + str(fib(liczba)))

import time
i = 0
while True:
    start_time = time.time()
    print("fib(" + str(i) + ") = " + str(fib(i)) + " (czas: " + str(round(time.time() - start_time, 2)) + ")")
    i += 1
