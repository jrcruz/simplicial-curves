import sys
top = 0
l = []
with open(sys.argv[1]) as file:
    for (index, row) in enumerate(map(lambda s: s.split(), file)):
        l.append(row)
        top = len(row)

for j in range(top):
    for k in range(len(l)):
        print(l[k][j], end=' ')
    print()
