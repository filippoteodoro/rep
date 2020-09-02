# Write a program to read through the mbox-short.txt and figure out who has sent the greatest number of mail messages.
# The program looks for 'From ' lines and takes the second word of those lines as the person who sent the mail.
# The program creates a Python dictionary that maps the sender's mail address to a count of the number of times they appear in the file.
# After the dictionary is produced, the program reads through the dictionary using a maximum loop to find the most prolific committer.

handle = open('mbox-short.txt')

mail_lst = list()

mail_dict = dict()

for line in handle:
    words = line.split()
    if len(words) < 1 or words[0] != 'From:':
        # if not line.startswith('From:'):
        continue
    mail_lst.append(words[1])

    # print(mail_lst)

    if words[1] in mail_dict:
        mail_dict[words[1]] = mail_dict[words[1]] + 1
    else:
        mail_dict[words[1]] = 1
#    print(words[1], mail_dict[words[1]])

# for email in mail_lst:
#    mail_dict[email] = mail_dict.get(email, 0) + 1

# print(mail_dict)

big_count = None
big_email = None
for key, value in mail_dict.items():
    if big_count is None or value > big_count:
        big_count = value
        big_email = key

print(big_email, big_count)

# cwen@iupui.edu 5
