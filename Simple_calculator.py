print("""==========
Prosty kalkulator
podaj typ operacji jaką chcesz wykonać
dozwolone operacje: + - * / ** max min
operacja "x" kończy działanie programu
następnie dodawaj liczby dla których chcesz dokonać obliczeń
==========""")

while True:
    operacja = input("Jaką operację chcesz wykonać? (+, -, *, /, **, min, max, x): ")
    if operacja.lower() == 'x':
        print("Do zobaczenia!")
        break
    liczby = []
    while True:
        x = input("Podaj liczbę (x, kończy wprowadzanie liczb): ")
        if x.lower() == 'x':
            break
        liczby.append(float(x))

    if operacja == '+':
        sum = 0
        for i in liczby:
            sum += i
        print("suma liczb w zbiorze " + str(liczby) + ": " + str(sum))
    elif operacja == '-':
        sum = 0
        for i in liczby:
            sum -= i
        print("róznica liczb w zbiorze " + str(liczby) + ": " + str(sum))
    elif operacja == '*':
        sum = 1
        for i in liczby:
            sum *= i
        print("iloraz liczb w zbiorze " + str(liczby) + ": " + str(sum))
    elif operacja == '/':
        sum = liczby[0]
        for i in range(1, len(liczby)): # liczby[1:-1]
            if liczby[i] == 0:
                continue
            sum /= liczby[i]
        print("iloczyn liczb w zbiorze " + str(liczby) + ": " + str(sum))
    elif operacja == '**':
        sum = liczby[0]
        for i in range(1, len(liczby)):
            sum **= liczby[i]
        print("potęga liczb w zbiorze " + str(liczby) + ": " + str(sum))
    elif operacja == 'max':
        print("największa liczba w zbiorze " + str(liczby) + ": " + str(max(liczby)))
    elif operacja == 'min':
        print("najmniejsza liczba w zbiorze " + str(liczby) + ": " + str(min(liczby)))
    else:
        print("nie umiem wykonać operacji \"" + operacja + "\" :(")
