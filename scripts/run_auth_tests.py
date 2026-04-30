#!/usr/bin/env python3
import urllib.request, json, sys
base='http://127.0.0.1:5000'

def post(path, data):
    req = urllib.request.Request(base+path, data=json.dumps(data).encode('utf-8'), headers={'Content-Type':'application/json'})
    return urllib.request.urlopen(req)

try:
    u = 'ci_test_user'
    p = 'Secr3t!'
    r = post('/register', {'username':u, 'password':p})
    print('/register', r.status)
    r = post('/login', {'username':u, 'password':p})
    print('/login', r.status)
    print('OK')
    sys.exit(0)
except Exception as e:
    print('FAILED', e)
    try:
        import urllib.error
        if isinstance(e, urllib.error.HTTPError):
            print(e.read().decode())
    except Exception:
        pass
    sys.exit(2)
