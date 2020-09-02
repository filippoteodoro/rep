fh = open('mbox-short.txt')

count = 0

for line in fh:
    line.rstrip()
    words = line.split()
    if len(words) < 1 or words[0] != 'From':
        continue
    count += 1
    print(words[1])

print("There were", count, "lines in the file with From as the first word")

