# Use the file name mbox-short.txt as the file name
fname = input("Enter file name: ")
fh = open(fname)
count = 0
total = 0
for line in fh:
    if not line.startswith("X-DSPAM-Confidence:"):
        continue
    s_line = line.rstrip()
    ipos = s_line.find('0')
    spam = float(s_line[ipos:])
    total = total + spam
    count += 1

average = str(total/count)

print('Average spam confidence: ' +average)

