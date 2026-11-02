"""Download and cache prices. Never fills missing prices."""
import sqlite3, pandas as pd, numpy as np
DB = "analyzer.db"

def _db():
    c = sqlite3.connect(DB)
    c.execute("create table if not exists px(t text, d text, v real, primary key(t,d))")
    c.execute("create table if not exists pulled(t text primary key, ts text)")
    c.execute("create table if not exists portfolios(name text primary key, body text)")
    return c

def _download(t):
    import yfinance as yf
    df = yf.download(t, period="max", auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {t}")
    s = df["Adj Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()

def series(t, use_cache=False):
    """Adjusted close for one ticker. Re-downloads in full when cache is older than 1 trading day."""
    c = _db()
    row = c.execute("select ts from pulled where t=?", (t,)).fetchone()
    stale = row is None or pd.Timestamp(row[0]) < pd.Timestamp.today().normalize() - pd.offsets.BDay(1)
    if row is None or (stale and not use_cache):
        s = _download(t)
        c.execute("delete from px where t=?", (t,))
        c.executemany("insert into px values(?,?,?)", [(t, d.strftime("%Y-%m-%d"), float(v)) for d, v in s.items()])
        c.execute("insert or replace into pulled values(?,?)", (t, pd.Timestamp.today().strftime("%Y-%m-%d")))
        c.commit()
    rows = c.execute("select d,v from px where t=? order by d", (t,)).fetchall()
    pulled = c.execute("select ts from pulled where t=?", (t,)).fetchone()[0]
    return pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows])), pulled

def benchmark(use_cache=False):
    try:
        return ("^SP500TR",) + series("^SP500TR", use_cache)
    except Exception:
        return ("SPY",) + series("SPY", use_cache)

def risk_free(index, use_cache=False):
    """Daily T-bill rate from FRED DTB3. Carried forward from the last published rate on days FRED has none."""
    import os, io, urllib.request
    p = "dtb3.csv"
    if not (use_cache and os.path.exists(p)):
        open(p, "wb").write(urllib.request.urlopen("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3", timeout=30).read())
    f = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
    f = pd.to_numeric(f, errors="coerce").dropna()
    y = f.reindex(f.index.union(index)).ffill().reindex(index)
    if y.isna().any():
        raise ValueError("Risk-free series does not cover the full window")
    return (1 + y / 100) ** (1 / 252) - 1

def load(tickers, start, end, use_cache=False):
    """Returns prices (common window, gap days dropped), benchmark name, pull dates, info about dropped days."""
    px, pulled = {}, {}
    for t in tickers:
        px[t], pulled[t] = series(t, use_cache)
    bname, bs, bp = benchmark(use_cache)
    px["__bench__"], pulled[bname] = bs, bp
    raw = pd.DataFrame(px)
    first = {c: raw[c].first_valid_index() for c in raw}
    lim = max(first, key=first.get)
    req = pd.Timestamp(start)
    window_start = max(first[lim], req)
    limited = "the requested start date" if req >= first[lim] else lim
    df = raw.loc[window_start:end]
    n_all = len(df)
    df = df.dropna()
    return df, bname, pulled, {"window_start": window_start, "window_end": df.index[-1], "limited_by": limited, "dropped_days": n_all - len(df)}
