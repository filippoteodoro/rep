fh = open('romeo.txt')  # doesn't store the values
lst = list()

for line in fh:
    line = line.rstrip()
    line = line.split()
    for words in line:
        if words not in lst:
            lst.append(words)

lst.sort()
print(lst)




