import os
import sys
import re

c_file = open("../cfile.csv", 'w')
s_file = open("../sfile.csv", 'w')
all_file = open("../allfile.csv", 'w')

for file in os.listdir(sys.argv[1]):
    result = re.findall(R".*c([0-9.]+)-s([0-9.]+).*\(([0-9.]+)\).*", file)
    if len(result) != 0:
        (c, s, ent) = result[0]
        c_file.write(c + ',' + ent + '\n')
        s_file.write(s + ',' + ent + '\n')
        all_file.write(c + ',' + s + ',' + ent + '\n')

c_file.close()
s_file.close()
all_file.close()

