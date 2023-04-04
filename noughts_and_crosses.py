import random
from os import system, name

plansza = [[" " for i in range(3)] for j in range(3)]

def wyczyscKonsole():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

def rysujWierszPusty():
    print("-------------")

def rysujWiersz(wiersz):
    print("|", wiersz[0], "|", wiersz[1], "|", wiersz[2], "|")

def rysujPlansze(plansza):
    wyczyscKonsole()
    for i in plansza:
        rysujWierszPusty()
        rysujWiersz(i)
    rysujWierszPusty()

def ruchKomutera(plansza, drukujInformacje = True):
    if drukujInformacje:
        print("ruch komputera!")
    ruch = random.randint(0, 8)
    wiersz = ruch // 3
    kolumna = ruch % 3
    if plansza[wiersz][kolumna] == " ":
        plansza[wiersz][kolumna] = "X"
    else:
        ruchKomutera(plansza, False)

def czyWygralKtosWiersz(plansza):
    for wiersz in plansza:
        if " " in wiersz:
            continue
        if "X" in wiersz and "O" in wiersz:
            continue
        if wiersz[0] == "X":
            return True, "komputer"
        else:
            return True, "uzytkownik"
    return False,

def czyWygralKtosKolumna(plansza):
    for kolumna in range(3):
        znakiKolumna = []
        for wiersz in range(3):
            znakiKolumna.append(plansza[wiersz][kolumna])
        if " " in znakiKolumna:
            continue
        if "X" in znakiKolumna and "O" in znakiKolumna:
            continue
        if znakiKolumna[0] == "X":
            return True, "komputer"
        else:
            return True, "uzytkownik"
    return False,

def czyWygralKtosPrzekatna(plansza):
    znaki = []
    for i in range(3):
        znaki.append(plansza[i][i])
    if "X" in znaki and "O" not in znaki and " " not in znaki:
        return True, "komputer"
    if "O" in znaki and "X" not in znaki and " " not in znaki:
        return True, "uzytkownik"
    return False,

def czyWygralKtosPrzekatna2(plansza):
    znaki = []
    for i in range(3):
        znaki.append(plansza[i][2-i])
    if "X" in znaki and "O" not in znaki and " " not in znaki:
        return True, "komputer"
    if "O" in znaki and "X" not in znaki and " " not in znaki:
        return True, "uzytkownik"
    return False,

def czyWygralKtos(plansza):
    wygrana = czyWygralKtosWiersz(plansza)
    if wygrana[0]:
        return wygrana
    wygrana = czyWygralKtosKolumna(plansza)
    if wygrana[0]:
        return wygrana
    wygrana = czyWygralKtosPrzekatna(plansza)
    if wygrana[0]:
        return wygrana
    wygrana = czyWygralKtosPrzekatna2(plansza)
    if wygrana[0]:
        return wygrana
    return False,

def ruchUzytkownika(plansza):
    ruch = input("wprowadz swój ruch (X Y): ")
    ruchTab = ruch.split(" ")
    wiersz = int(ruchTab[0])
    kolumna = int(ruchTab[1])
    if plansza[wiersz][kolumna] == " ":
        plansza[wiersz][kolumna] = "O"
    else:
        print("nie mozna wykonac tego ruchu")
        ruchUzytkownika(plansza)

def czyRemis(plansza):
    for wiersz in range(3):
        for kolumna in range(3):
            if plansza[wiersz][kolumna] == " ":
                return False
    return True

i = 0
while True:
    if i % 2 == 0:
        ruchKomutera(plansza)
    else:
        ruchUzytkownika(plansza)
    rysujPlansze(plansza)
    wynik = czyWygralKtos(plansza)
    if wynik[0] == True:
        print(wynik[1], "wygrał!")
        break
    if czyRemis(plansza):
        print("remis!")
        break
    i += 1
    
