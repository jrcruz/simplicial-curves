import sys
import re

all_file = open("allfile.csv", 'w')

for file in open(sys.argv[1]):
    result = re.findall(R".*c([0-9.]+)-s([0-9.]+).*\(([0-9.]+)\).*", file)
    if len(result) != 0:
        (c, s, ent) = result[0]
        all_file.write(c + ',' + s + ',' + ent + '\n')

all_file.close()
