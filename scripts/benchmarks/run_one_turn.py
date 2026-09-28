#!/usr/bin/env python3
import json, sys, time, urllib.request
base = (sys.argv[1] if len(sys.argv)>1 else 'http://192.168.0.153:8091').rstrip('/')
q = sys.argv[2] if len(sys.argv)>2 else '我记得有次晚上在河北户外的婚礼仪式舞台前拍了留影，帮我找到这张照片对应的那次经历？'
def call(path, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'} if data else {})
    with urllib.request.urlopen(req,timeout=900) as r:return json.loads(r.read().decode())
cid='one-'+str(int(time.time()))
r=call('/api/assistant/turn',{'message':q,'scope_id':'album_ca0cc0ddda3a','conversation_id':cid,'viewer_id':'owner','include_debug':True})
if r.get('status') in {'running','pending'}:
 tid=r['turn_id']
 while True:
  s=call('/api/assistant/turn/'+str(tid))
  if str(s.get('status','')).lower() in {'complete','completed','done','success'}: r=s.get('result') or {};break
  if str(s.get('status','')).lower() in {'failed','error','cancelled','canceled'}: r=s;break
  time.sleep(.5)
print(json.dumps(r,ensure_ascii=False,indent=2))
