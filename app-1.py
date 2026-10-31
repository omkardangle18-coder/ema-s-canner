#!/usr/bin/env python3
"""
EMA 35 / EMA 200 crossover scanner (3-minute timeframe).

Standard library only (Python 3.9+). Only EMA 35 and EMA 200 are used.

Run:
    python3 app.py
Then open http://127.0.0.1:8000

Environment variables (all optional unless noted):
    TWELVEDATA_API_KEY   required for forex / metals (free key at twelvedata.com)
    SCAN_INTERVAL_SEC    seconds between automatic scans        (default 60)
    MAX_SIGNAL_AGE_MIN   confirmed signals older than this are removed (default 30)
    HOST                 bind address                           (default 0.0.0.0)
    PORT                 HTTP port                              (default 8000)
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---- Strategy (fixed) -------------------------------------------------------
FAST_PERIOD = 35
SLOW_PERIOD = 200
TIMEFRAME_MIN = 3
CANDLES_NEEDED = 600  # ~30 hours of 3m candles; EMA 200 is well converged by then

# ---- Configuration ----------------------------------------------------------
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL_SEC", "60"))
MAX_SIGNAL_AGE_MIN = int(os.getenv("MAX_SIGNAL_AGE_MIN", "30"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Crypto: Binance public klines (no API key needed)
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
# Forex / metals: Twelve Data format (BASE/QUOTE). Free tier allows 8 requests/min,
# so keep this list short or raise SCAN_INTERVAL_SEC.
FOREX_SYMBOLS = ["XAU/USD", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]

HTTP_TIMEOUT = 10
TD_SPACING_SEC = 8  # spacing between Twelve Data calls
INDEX_HTML = Path(__file__).with_name("index.html")

# ---- Shared state (guarded by state_lock) -----------------------------------
state_lock = threading.Lock()
status = {}        # display symbol -> status dict
known = {}        # display symbol -> latest confirmed signal dict, or None
last_scan_ms = None
refresh_event = threading.Event()


def now_ms_():
    return int(time.time() * 1000)


# ---- Data sources -----------------------------------------------------------
def http_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ema-scanner/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_binance(symbol):
    """Binance spot klines. Returns candles sorted oldest -> newest."""
    q = urllib.parse.urlencode({"symbol": symbol, "interval": "3m", "limit": CANDLES_NEEDED})
    rows = http_json(f"https://api.binance.com/api/v3/klines?{q}")
    if isinstance(rows, dict):  # error payload such as {"code":..., "msg":...}
        raise RuntimeError(rows.get("msg", "Binance returned an error"))
    # Row: [openTime, open, high, low, close, volume, closeTime, ...]
    # closeTime is the last millisecond of the candle, so +1 gives the candle end.
    return [
        {"open_ms": int(r[0]), "close_ms": int(r[6]) + 1, "close": float(r[4])}
        for r in rows
    ]


def fetch_twelvedata(symbol):
    """Twelve Data time series (forex / metals). Returns candles oldest -> newest."""
    if not TWELVEDATA_KEY:
        raise RuntimeError("TWELVEDATA_API_KEY not set")
    q = urllib.parse.urlencode({
        "symbol": symbol,
        "interval": "3min",
        "outputsize": CANDLES_NEEDED,
        "timezone": "UTC",
        "apikey": TWELVEDATA_KEY,
    })
    data = http_json(f"https://api.twelvedata.com/time_series?{q}")
    if data.get("status") == "error" or "values" not in data:
        raise RuntimeError(data.get("message", "Twelve Data returned an error"))
    candles = []
    for v in data["values"]:
        dt = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        open_ms = int(dt.timestamp() * 1000)
        candles.append({
            "open_ms": open_ms,
            "close_ms": open_ms + TIMEFRAME_MIN * 60_000,
            "close": float(v["close"]),
        })
    candles.sort(key=lambda c: c["open_ms"])
    return candles


# ---- Strategy logic ---------------------------------------------------------
def ema(values, period):
    """Standard EMA: seeded with the SMA of the first `period` values."""
    out = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    prev = sum(values[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def latest_confirmed_cross(candles, now_ms):
    """
    Find the most recent EMA 35 / EMA 200 crossover on CLOSED 3m candles only.
    A candle still forming (close_ms > now) is ignored, so every signal is confirmed.
    Returns (cross_or_None, close_ms_of_last_closed_candle).
    """
    closed = [c for c in candles if c["close_ms"] <= now_ms]
    if len(closed) < SLOW_PERIOD + 2:
        raise RuntimeError(f"not enough closed candles ({len(closed)})")

    closes = [c["close"] for c in closed]
    fast = ema(closes, FAST_PERIOD)
    slow = ema(closes, SLOW_PERIOD)

    found = None
    for i in range(1, len(closed)):
        if fast[i - 1] is None or slow[i - 1] is None:
            continue
        prev_diff = fast[i - 1] - slow[i - 1]
        diff = fast[i] - slow[i]
        if prev_diff <= 0 < diff:
            side = "BUY"
        elif prev_diff >= 0 > diff:
            side = "SELL"
        else:
            continue
        found = {"side": side, "candle": closed[i], "ema_fast": fast[i], "ema_slow": slow[i]}
    return found, closed[-1]["close_ms"]


def process(symbol, exchange, fetcher):
    display = symbol.replace("/", "")
    now = now_ms_()
    try:
        candles = fetcher(symbol)
        cross, last_close = latest_confirmed_cross(candles, now)
        sig = None
        if cross:
            close_ms = cross["candle"]["close_ms"]
            sig = {
                "id": f"{display}-{cross['side']}-{close_ms}",
                "symbol": display,
                "side": cross["side"],
                "timeframe": "3M",
                "crossover_ms": close_ms,   # confirmed when the candle closed
                "ema35": round(cross["ema_fast"], 8),
                "ema200": round(cross["ema_slow"], 8),
                "exchange": exchange,
            }
        st = {
            "symbol": display, "exchange": exchange, "ok": True, "error": None,
            "last_candle_close_ms": last_close, "checked_ms": now,
        }
        return display, st, sig
    except Exception as e:  # network, rate limit, bad payload, missing key, etc.
        st = {
            "symbol": display, "exchange": exchange, "ok": False,
            "error": str(e)[:200], "last_candle_close_ms": None, "checked_ms": now,
        }
        return display, st, None


def scan_all():
    global last_scan_ms
    results = [process(s, "Binance Spot", fetch_binance) for s in CRYPTO_SYMBOLS]
    for i, s in enumerate(FOREX_SYMBOLS):
        if i:
            time.sleep(TD_SPACING_SEC)
        results.append(process(s, "Twelve Data (FX/metals)", fetch_twelvedata))

    with state_lock:
        for display, st, sig in results:
            status[display] = st
            if st["ok"]:
                known[display] = sig  # failed fetches keep the previous signal until it ages out
        last_scan_ms = now_ms_()


def scanner_loop():
    while True:
        started = time.time()
        try:
            scan_all()
        except Exception as e:
            print(f"[scan error] {e}")
        remaining = max(1.0, SCAN_INTERVAL - (time.time() - started))
        refresh_event.wait(timeout=remaining)  # wakes early on manual refresh
        refresh_event.clear()


def build_state():
    now = now_ms_()
    with state_lock:
        sigs = []
        for sym, sig in list(known.items()):
            if sig is None:
                continue
            if now - sig["crossover_ms"] > MAX_SIGNAL_AGE_MIN * 60_000:
                known[sym] = None  # remove stale signal
                continue
            sigs.append(dict(sig))
        sigs.sort(key=lambda s: s["crossover_ms"], reverse=True)
        return {
            "server_time_ms": now,
            "last_scan_ms": last_scan_ms,
            "scan_interval_sec": SCAN_INTERVAL,
            "max_signal_age_min": MAX_SIGNAL_AGE_MIN,
            "signals": sigs,
            "symbols": {k: dict(v) for k, v in status.items()},
        }


# ---- HTTP server ------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._send(200, build_state())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path == "/api/refresh":
            refresh_event.set()
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    if not TWELVEDATA_KEY:
        print("[warn] TWELVEDATA_API_KEY is not set: forex/metals will show as unavailable")
    threading.Thread(target=scanner_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"EMA 35/200 3M scanner running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
