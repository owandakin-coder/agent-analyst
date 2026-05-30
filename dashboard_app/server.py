"""
QuantPulse — Local server
Serves index.html + proxies Alpaca Paper Trading API & Yahoo Finance
Keys loaded from ../.env — never exposed to the browser.
Run: python dashboard_app/server.py
"""
import http.server, json, os, sys, urllib.request, urllib.error, threading, gzip
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as HTTPServer
from urllib.parse import urlparse, parse_qs

ROOT_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)

from control_plane import apply_control_action, control_status_summary, dispatch_trade_workflow

PORT = 7788
ROOT = os.path.dirname(os.path.abspath(__file__))   # dashboard_app/
ENV_PATH = os.path.join(ROOT, '..', '.env')          # agent analyst/.env

MIME = {
    'html': 'text/html; charset=utf-8',
    'js':   'application/javascript',
    'css':  'text/css',
    'json': 'application/json',
    'png':  'image/png',
    'ico':  'image/x-icon',
    'svg':  'image/svg+xml',
}

# ── Load .env ────────────────────────────────────────────────────────────────
def load_env(path):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env

ENV = load_env(ENV_PATH)
ALPACA_KEY    = ENV.get('ALPACA_API_KEY', '')
ALPACA_SECRET = ENV.get('ALPACA_SECRET_KEY', '')
ALPACA_BASE   = ENV.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets/v2').rstrip('/')

print(f'Alpaca: {ALPACA_BASE}', flush=True)
print(f'Key   : {ALPACA_KEY[:8]}...', flush=True)

# ── Alpaca helper ─────────────────────────────────────────────────────────────
def alpaca_get(path, params=''):
    url = f'{ALPACA_BASE}{path}'
    if params:
        url += '?' + params
    headers = {
        'APCA-API-KEY-ID':     ALPACA_KEY,
        'APCA-API-SECRET-KEY': ALPACA_SECRET,
        'Accept':              'application/json',
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=12) as r:
        raw = r.read()
        if r.info().get('Content-Encoding') == 'gzip':
            raw = gzip.decompress(raw)
        return json.loads(raw)

# ── Yahoo Finance helper ──────────────────────────────────────────────────────
YF_HEADERS = {
    'User-Agent':      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept':          'application/json',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer':         'https://finance.yahoo.com/',
}

def fetch_yf(symbol):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1d&interval=1d'
    req = urllib.request.Request(url, headers=YF_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = r.read()
            if r.info().get('Content-Encoding') == 'gzip':
                raw = gzip.decompress(raw)
            data = json.loads(raw)
        meta = data['chart']['result'][0]['meta']
        prev = (meta.get('regularMarketPreviousClose')
             or meta.get('previousClose')
             or meta.get('chartPreviousClose')
             or meta.get('regularMarketPrice', 0))
        return {'symbol': symbol, 'price': meta.get('regularMarketPrice', 0), 'prevClose': prev}
    except Exception as e:
        return {'symbol': symbol, 'error': str(e)}

def fetch_yf_many(symbols_str):
    symbols  = [s.strip() for s in symbols_str.split(',') if s.strip()]
    results  = [None] * len(symbols)
    def worker(i, s): results[i] = fetch_yf(s)
    threads = [threading.Thread(target=worker, args=(i, s)) for i, s in enumerate(symbols)]
    for t in threads: t.start()
    for t in threads: t.join()
    return results

# ── HTTP Handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass   # quiet

    def send_cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200); self.send_cors(); self.end_headers()

    def reply_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_cors()
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get('Content-Length', '0') or 0)
        raw = self.rfile.read(length) if length else b'{}'
        if not raw:
            return {}
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        qs     = parse_qs(parsed.query)

        # ── /api/account — Alpaca account summary ───────────────────────────
        if path == '/api/account':
            try:
                data = alpaca_get('/account')
                self.reply_json({
                    'equity':         float(data.get('equity', 0)),
                    'last_equity':    float(data.get('last_equity', 0)),
                    'cash':           float(data.get('cash', 0)),
                    'buying_power':   float(data.get('buying_power', 0)),
                    'portfolio_value':float(data.get('portfolio_value', 0)),
                })
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        # ── /api/positions — Alpaca positions ───────────────────────────────
        if path == '/api/positions':
            try:
                positions = alpaca_get('/positions')
                result = []
                for p in positions:
                    result.append({
                        'symbol':       p['symbol'],
                        'qty':          float(p['qty']),
                        'avg_entry':    float(p.get('avg_entry_price', 0)),
                        'current_price':float(p.get('current_price', 0)),
                        'market_value': float(p.get('market_value', 0)),
                        'unrealized_pl':float(p.get('unrealized_pl', 0)),
                        'unrealized_plpc': float(p.get('unrealized_plpc', 0)),
                        'change_today': float(p.get('change_today', 0)),
                    })
                self.reply_json(result)
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        # ── /api/orders — Alpaca trade history ──────────────────────────────
        if path == '/api/orders':
            try:
                orders = alpaca_get('/orders', 'status=closed&limit=50&direction=desc')
                result = []
                for o in orders:
                    if o.get('filled_at') and o.get('filled_avg_price'):
                        result.append({
                            'id':     o['id'],
                            'symbol': o['symbol'],
                            'side':   o['side'],
                            'qty':    float(o.get('filled_qty', o.get('qty', 0))),
                            'price':  float(o.get('filled_avg_price', 0)),
                            'time':   o.get('filled_at', ''),
                            'type':   o.get('type', ''),
                        })
                self.reply_json(result)
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        # ── /api/history — Alpaca portfolio equity curve ─────────────────────
        if path == '/api/history':
            period    = qs.get('period',    ['1M'])[0]
            timeframe = qs.get('timeframe', ['1D'])[0]
            try:
                data = alpaca_get('/account/portfolio/history',
                                  f'period={period}&timeframe={timeframe}&intraday_reporting=market_hours')
                timestamps = data.get('timestamp', [])
                equity     = data.get('equity', [])
                self.reply_json({'timestamps': timestamps, 'equity': equity})
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        # ── /api/quotes — Yahoo Finance fallback ─────────────────────────────
        if path == '/api/quotes':
            symbols = qs.get('symbols', ['AAPL,NVDA,MSFT,JPM,META'])[0]
            self.reply_json({'quotes': fetch_yf_many(symbols)})
            return

        if path == '/api/control':
            try:
                self.reply_json(control_status_summary())
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        # ── Static files ──────────────────────────────────────────────────────
        if path == '/': path = '/index.html'
        file_path = os.path.join(ROOT, path.lstrip('/'))
        if not os.path.isfile(file_path):
            self.send_response(404); self.end_headers(); return
        ext  = file_path.rsplit('.', 1)[-1].lower()
        mime = MIME.get(ext, 'application/octet-stream')
        with open(file_path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/control':
            try:
                payload = self.read_json()
                action = str(payload.get('action', '')).strip().lower()
                actor = str(payload.get('actor', 'dashboard_local')).strip() or 'dashboard_local'
                if not action:
                    self.reply_json({'error': 'action is required'}, 400)
                    return
                self.reply_json(apply_control_action(action, actor=actor))
            except ValueError as e:
                self.reply_json({'error': str(e)}, 400)
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        if path == '/api/control/run_once':
            try:
                payload = self.read_json()
                actor = str(payload.get('actor', 'dashboard_local')).strip() or 'dashboard_local'
                self.reply_json(dispatch_trade_workflow(actor=actor), 202)
            except Exception as e:
                self.reply_json({'error': str(e)}, 500)
            return

        self.reply_json({'error': 'Not found'}, 404)


if __name__ == '__main__':
    server = HTTPServer(('', PORT), Handler)
    print(f'QuantPulse server running on http://localhost:{PORT}', flush=True)
    server.serve_forever()
