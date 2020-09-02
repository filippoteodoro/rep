# To run this, download the BeautifulSoup zip file
# http://www.py4e.com/code3/bs4.zip
# and unzip it in the same directory as this file

from urllib import request
from bs4 import BeautifulSoup
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

html = request.urlopen('http://py4e-data.dr-chuck.net/comments_320288.html', context=ctx).read()
soup = BeautifulSoup(html, "html.parser")

sum = 0
# Retrieve all of the span tags
tags = soup('span')
for tag in tags:
    sum = sum + int(tag.contents[0])

print(sum)
