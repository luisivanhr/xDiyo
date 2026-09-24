"""Inspect assets against the current report renderer and write a visual sample."""
import ast
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

p = Path(__file__).resolve().parent
root = p.parents[2]
tree = ast.parse((root/'src/xdiyo_analytics/reporting/teams.py').read_text(encoding='utf-8'))
# Run the exact renderer helper source without importing optional pandas.
nodes = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and not (isinstance(n,ast.Import) and n.names[0].name=='pandas')] + [n for n in tree.body if isinstance(n,ast.FunctionDef)]
ns = {}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'teams.py','exec'),ns)
c = json.loads((p/'catalog.json').read_text(encoding='utf-8'))
bad, cards = [], []
good = 0
fm = Counter()
asset_bytes = 0
raster_svgs = 0
for key,item in c.items():
    if 'badge_path' not in item:
        continue
    try:
        q = p/item['badge_path']
        assert hashlib.sha256(q.read_bytes()).hexdigest() == item['sha256']
        src = ns['_badge'](q)
        asset_bytes += q.stat().st_size
        if q.suffix == '.svg':
            svg = ET.fromstring(q.read_bytes())
            raster_svgs += int(any(node.tag.endswith('image') for node in svg.iter()))
        else:
            Image.open(q).verify()
        good += 1
        fm[item['format']] += 1
        if key in ['42','17','2672','2692','2817','2888','2953','2352','8','11','1657','2859','1652','2671','1664','2567']:
            cards.append(f'<figure><img src="{src}"><figcaption>{html.escape(item["name"])}</figcaption></figure>')
    except Exception as e:
        bad.append({'id':key,'name':item['name'],'error':str(e)})
summary = {'checked':good+len(bad),'accepted':good,'formats':dict(fm),'rejected':bad}
print(json.dumps(summary))
(p/'renderer_compatibility.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(p/'integrity.json').write_text(json.dumps({'checked':good+len(bad),'formats':dict(fm),'bytes':asset_bytes,'svg_with_image_elements':raster_svgs,'errors':bad},indent=2)+'\n',encoding='utf-8')
(p/'sample.html').write_text('<!doctype html><meta charset=utf-8><title>Team badge inspection</title><style>body{font:15px system-ui;background:#f5f7fa;color:#172334}main{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}figure{background:white;border:1px solid #ddd;border-radius:8px;margin:0;text-align:center;padding:16px}img{height:96px;width:130px;object-fit:contain}figcaption{margin-top:8px}</style><h1>Acquired team badges</h1><main>'+''.join(cards)+'</main>',encoding='utf-8')
