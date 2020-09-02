import re

handle = open('regex_sum_320286.txt')
sum = 0

for line in handle:
    numbers = re.findall('[0-9]+', line)
    if not numbers:
        continue
    else:
        for number in numbers:
            sum += int(number)

print(sum)