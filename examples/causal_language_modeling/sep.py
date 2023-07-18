import sys

for aline in sys.stdin:
    aline = aline.strip()
    aline = aline.replace('\'', '\"')
    print(aline)
