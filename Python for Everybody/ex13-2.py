import urllib.request, urllib.parse, urllib.error
import json
import ssl

# Ignore SSL certificate errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# South Federal University

api_key = 42
serviceurl = 'http://py4e-data.dr-chuck.net/json?'

while True:
    address = input('Enter location: ')
    if len(address) < 1:
        break
    parms = dict()
    parms['address'] = address
    parms['key'] = api_key

    url = serviceurl + urllib.parse.urlencode(parms)  # encodes addresses in URL format

    print('Retrieving', url)
    data = urllib.request.urlopen(url, context=ctx).read().decode()
    # print(data)
    print('Retrieved', len(data), 'characters')

    try:
        js = json.loads(data)
    except:
        js = None
    # print(js)

    if not js or 'status' not in js or js['status'] != 'OK':
        print('==== Failure To Retrieve ====')
        continue

    # print(len(js['results']))
    # print(json.dumps(js))
    place_id = js['results'][0]['place_id']
    print('Place id ' + place_id)
