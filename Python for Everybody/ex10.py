# 10.2 Write a program to read through the mbox-short.txt and figure out the distribution by hour of the day for each
# of the messages. You can pull the hour out from the 'From' line by finding the time and then splitting the string a
# second time using a colon. Once you have accumulated the counts for each hour, print out the counts, sorted by hour as shown below.

handle = open('mbox-short.txt',)
words_lst = list()
hours_lst = list()
hours_dict = dict()

for line in handle:
    words = line.split()
    if not line.startswith('From '):
        continue
    words_lst.append(words[5])

# print(words_lst)

for time in words_lst:
    hours = time.split(':')
    hours_lst.append(hours[0])
    if hours[0] in hours_dict:
        hours_dict[hours[0]] = hours_dict[hours[0]] + 1
    else:
        hours_dict[hours[0]] = 1

# print(hours_dict)
tuples_lst = []

for key, value in hours_dict.items():
    tuples_lst.append((key, value))

# tuples_lst.sort()
# print(lst)

for key, value in sorted(tuples_lst):
    print(key, value)