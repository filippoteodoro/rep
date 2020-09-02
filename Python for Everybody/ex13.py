from urllib import request
import xml.etree.ElementTree as ET
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

html = request.urlopen('http://py4e-data.dr-chuck.net/comments_320290.xml', context=ctx).read()
tree = ET.fromstring(html)
lst = tree.findall('comments/comment')

sum = 0

for item in lst:
    count = int(item.find('count').text)
    sum = count + sum

print(sum)