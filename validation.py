walidatorMaleLitery = [chr(i) for i in range(97, 123)]
walidatorWielkiLitery = [i.upper() for i in walidatorMaleLitery]
walidatorCyfry = [str(i) for i in range(0, 10)]
walidatorZnakiSpecjalne = ["$", "#", "@"]
walidatory = {
    "walidatorMaleLitery": walidatorMaleLitery,
    "walidatorWielkiLitery": walidatorWielkiLitery,
    "walidatorCyfry": walidatorCyfry,
    "walidatorZnakiSpecjalne": walidatorZnakiSpecjalne
}

def walidacjaHasla(haslo):
    if len(haslo) < 6 or len(haslo) > 12:
        return False, "hasło zawiera niepoprawną długość"
    for nazwa, walidator in walidatory.items():
        walid = False
        for i in walidator:
            if i in haslo:
                walid = True
                break
        if walid == False:
            return False, nazwa
    return True,


haslo = input("Wprowadź haslo do walidacji: ")
walidacja = walidacjaHasla(haslo)
if walidacja[0]:
    print("Hasło jest poprawne")
else:
    print("Hasło niepoprawne,", walidacja[1])
