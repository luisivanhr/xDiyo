"""Token-protected loopback transport for already-computed spatial report data."""
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
import atexit
import json
import secrets
import threading


class SpatialStore:
    def __init__(self):
        self.data = {}
        self.ids = {}

    def register(self, data):
        key = self.ids.get(id(data))
        if key is None:
            key = secrets.token_urlsafe(18)
            self.ids[id(data)] = key
            self.data[key] = data
        return key

    def fetch(self, request):
        return self.data[request['id']].panel_pair(request['fixture'], request['map'])


@dataclass
class LiveReport:
    server: object
    thread: object
    store: SpatialStore
    token: str
    document: str = ''
    closed: bool = False

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/{self.token}/'

    def close(self):
        if not self.closed:
            self.closed = True
            self.server.shutdown()
            self.server.server_close()
            self.store.data.clear()
            self.store.ids.clear()
            self.document = ''


_handles = {}

def live_report(report):
    key = getattr(report, '_spatial_live_key', None)
    if key in _handles and not _handles[key].closed:
        return _handles[key]
    store, token = SpatialStore(), secrets.token_urlsafe(32)
    handle = None
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, data, code=200, content='application/json'):
            raw = data.encode() if isinstance(data, str) else json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type',content+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers(); self.wfile.write(raw)
        def authorized(self):
            host=f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host') == host and self.headers.get('Origin','http://'+host)=='http://'+host
        def do_GET(self):
            if not self.authorized() or urlparse(self.path).path != '/'+token+'/':
                return self.send({'error':'Not found'},404)
            self.send(handle.document,content='text/html')
        def do_POST(self):
            if not self.authorized() or urlparse(self.path).path != '/'+token+'/spatial' or not secrets.compare_digest(self.headers.get('X-Builder-Token',''),token):
                return self.send({'error':'Access denied'},403)
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 8192: raise ValueError('Invalid request size.')
                self.send(store.fetch(json.loads(self.rfile.read(size))))
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                self.send({'error':str(exc)},400)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True,name='spatial-report')
    handle=LiveReport(server,thread,store,token)
    handle.document=report.to_html(spatial_transport={'register':store.register,'url':handle.url+'spatial','token':token})
    thread.start()
    report._spatial_live_key=token
    _handles[token]=handle
    return handle


def close_report(report):
    handle=_handles.pop(getattr(report,'_spatial_live_key',None),None)
    if handle: handle.close()


@atexit.register
def _close_all():
    for handle in list(_handles.values()): handle.close()