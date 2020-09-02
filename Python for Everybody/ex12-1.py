from urllib import request
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

position = int(input('Enter position: ')) - 1
count = int(input('Enter count: ')) + 1
lists = []

for p in range(count):
    lists.append([])

loop = 0
times = 0
a = 0

while a <= position:
    lists[times].append('http://py4e-data.dr-chuck.net/known_by_Tori.html')
    a += 1

for list in lists:
    print(lists[times][position])
    html = request.urlopen(lists[times][position], context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    urls = soup('a')
    times += 1
    for url in urls:
        lists[times].append(url.get('href', None))
    if times < count:
        continue
    else:
        break
