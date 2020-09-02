from urllib import request
import json
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

data = request.urlopen('http://py4e-data.dr-chuck.net/comments_320291.json', context=ctx).read().decode()
info = json.loads(data)

# print('User count:', len(info['comments']))
# print(data)
# print(info)

sum = 0

for user in info['comments']:
    count = int(user['count'])
    sum = sum + count

print(sum)
