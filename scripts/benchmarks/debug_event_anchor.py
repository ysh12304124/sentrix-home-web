import backend.app as app
from backend.agent_runtime import tools
tools.bind_runtime(app.store, gamma=app.gamma)
import re, json
for q in __import__('sys').argv[1:]:
 print('QUERY',q, 'RESULT', tools._event_keyword_anchor(q, app.store, 'album_ca0cc0ddda3a'))
 ignored = {'我们','我和','家人','一起','去参观','参观','旅行','那次','在哪里','超级','的','那次旅行'}
 terms=[]
 for length in (4,3,2):
  for m in re.finditer(rf'[\u4e00-\u9fff]{{{length}}}',q):
   t=m.group(0)
   if t not in ignored and not any(x in t for x in ignored): terms.append(t)
 terms=list(dict.fromkeys(terms)); print('TERMS',terms)
 out=[]
 for raw in app.store.connection.execute("select id,title,summary from events where scope_id=?",('album_ca0cc0ddda3a',)):
  row=dict(raw); hay=' '.join(str(row.get(k) or '') for k in ('title','summary'))
  obs=app.store.connection.execute("select caption,activity,place,ocr_text from observations o join event_observations eo on eo.observation_id=o.id where eo.event_id=?",(row['id'],)).fetchall()
  hay+=' '+' '.join(' '.join(str(dict(item).get(k) or '') for k in ('caption','activity','place','ocr_text')) for item in obs)
  matched=[t for t in terms if t in hay]
  if '婚礼' in terms and '婚礼' not in matched and '婚' in hay: matched.append('婚礼')
  if any('船' in t for t in terms) and '船' not in matched and any(x in hay for x in ('游轮','滚装')): matched.append('船')
  if '婚礼' in matched and '户外' in matched: matched.append('__outdoor_wedding_pair__')
  if matched: out.append((len(set(matched)),row['id'],row['title'],matched))
 print('TOP', sorted(out,reverse=True)[:8])
 if '沙盘' in q:
  for raw in app.store.connection.execute("select e.id,e.title,e.summary from events e where e.scope_id=?",('album_ca0cc0ddda3a',)):
   row=dict(raw)
   obs=app.store.connection.execute("select caption,activity,place,ocr_text from observations o join event_observations eo on eo.observation_id=o.id where eo.event_id=?",(row['id'],)).fetchall()
   text=' '.join(str(row.get(k) or '') for k in ('title','summary'))+' '+' '.join(' '.join(str(dict(item).get(k) or '') for k in ('caption','activity','place','ocr_text')) for item in obs)
   if any(x in text for x in ('模型','沙盘','建筑模型')): print('MODEL_EVENT',row['id'],row['title'],text[:300])
