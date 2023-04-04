import random
import math

def generujPoziom(iloscRuchow):
    mozliwosci = list(range(random.randint(2, iloscRuchow)))
    ruch = mozliwosci[random.randint(0, len(mozliwosci) - 1)]
    return {
        "ruch": ruch,
        "mozliwosci": mozliwosci
    }

def generujLabirynt(poziom):
    labirynt = {}
    if poziom == 1:
        for i in range(5):
            labirynt[i] = generujPoziom(2)
    else:
        for i in range(8):
            labirynt[i] = generujPoziom(3)
    return labirynt

def policzMozliwosciPesymistyczne(labirynt):
    suma = 0
    for ruch in labirynt.values():
        suma += len(ruch["mozliwosci"])
    return suma

def wykonajRuch(poziom, ruch = 0):
    ruch += 1
    info = "wykonaj ruch: A - w lewo, D - w prawo"
    if len(poziom["mozliwosci"]) == 3:
        info += ", W - prosto"
    print(info)
    ruchUzytkownika = input('Gdzie chcesz iść?: ')
    if not ((ruchUzytkownika == "A" and poziom["ruch"] == 0)
            or (ruchUzytkownika == "D" and poziom["ruch"] == 1)
            or (ruchUzytkownika == "W" and poziom["ruch"] == 2)):
        print("zły ruch, próbuj dalej")
        return wykonajRuch(poziom, ruch)
    return ruch

def wyjdzZLabiryntu(labirynt, ruchyMaks):
    ruchyGracza = 0
    for decyzja in labirynt:
        print("\ndecyzja:", decyzja + 1)
        ruchyGracza += wykonajRuch(labirynt[decyzja])
        if ruchyGracza > ruchyMaks:
            break
    print()
    if ruchyGracza > ruchyMaks:
        print("umarłeś z głodu :(")
    else:
        print("brawo, udało Ci się wydostać z labiryntu :)")
    return ruchyGracza

poziom = int(input("podaj poziom trudności, 1 - łatwy, 2 - trudny: "))
labirynt = generujLabirynt(poziom)
ruchyOptymistyczne = 5 if poziom == 1 else 8
ruchyPesymistyczne = policzMozliwosciPesymistyczne(labirynt)
ruchySmierc = ruchyOptymistyczne + math.ceil((ruchyPesymistyczne - ruchyOptymistyczne) * 0.6)
ruchyGracza = wyjdzZLabiryntu(labirynt, ruchySmierc)

print("\n=== STATYSTYKI ===")
print("Wykonałeś ruchów:", ruchyGracza)
print("Minimalna ilość ruchów aby się wydostać:", ruchyOptymistyczne)
print("Śmiertelna ilość ruchów:", ruchySmierc + 1)
print("=== LABIRYNT ===")
print(labirynt)
