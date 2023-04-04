zbior = []

while True:
    x = input("Podaj liczbę do zbioru do posortowania (x, kończy wprowadzanie liczb): ")
    if x.lower() == 'x':
        break
    zbior.append(int(x))

print("wprowadzony zbiór: ", zbior)

# iteracja = 0
# for i in range(len(zbior)):
#    for j in range(len(zbior) - 1):
#        if zbior[j] > zbior[j + 1]:
#            zbior[j], zbior[j + 1] = zbior[j + 1], zbior[j]
#        iteracja += 1
# print("zbiór posortowany: ", zbior)
# print("przeprowadzonych iteracji:", iteracja)

iteracja = 0
zmieniono = True
while zmieniono:
    zmieniono = False
    for i in range(len(zbior) - 1):
        if zbior[i] > zbior[i + 1]:
            zbior[i], zbior[i + 1] = zbior[i + 1], zbior[i]
            zmieniono = True
        iteracja += 1
print("zbiór posortowany: ", zbior)
print("przeprowadzonych iteracji:", iteracja)
