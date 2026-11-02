import json, io, csv, pandas as pd
from flask import Flask, request, render_template, Response
import data, metrics
from portfolio import portfolio_returns
app = Flask(__name__)

def parse(form):
    rows = [(t.strip().upper(), p.strip()) for t, p in zip(form.getlist("t"), form.getlist("p")) if t.strip() or p.strip()]
    if not rows: raise ValueError("Enter at least one holding.")
    w = {}
    for t, p in rows:
        if not t: raise ValueError("A row has a percentage but no ticker.")
        if t in w: raise ValueError(f"Duplicate ticker: {t}.")
        try: v = float(p)
        except ValueError: raise ValueError(f"Allocation for {t} is not a number.")
        if v < 0: raise ValueError(f"Allocation for {t} is negative.")
        w[t] = v
    if abs(sum(w.values()) - 100) > 0.01: raise ValueError(f"Allocations total {sum(w.values()):.2f}%, not 100%.")
    return {t: v / 100 for t, v in w.items()}

def run(w, start, end, use_cache):
    px, bname, pulled, info = data.load(list(w), start, end, use_cache)
    r = px.pct_change().dropna()
    rm = r.pop("__bench__")
    rf = data.risk_free(px.index, use_cache).reindex(r.index)
    rp = portfolio_returns(px.drop(columns="__bench__"), w)
    m, b, verdict, tests = metrics.analyze(rp, rm, rf)
    return {"m": m, "b": b, "verdict": verdict, "tests": tests, "bname": bname, "pulled": pulled, "info": info,
            "days": len(r), "hold": metrics.holdings(r, rp, rm, rf, w), "n": len(r)}

@app.route("/", methods=["GET", "POST"])
def index():
    f = request.form; today = pd.Timestamp.today().normalize()
    ctx = {"rows": [("", "")] * 5, "err": None, "res": None,
           "start": (today - pd.DateOffset(years=10)).strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d"),
           "cache": False, "saved": [r[0] for r in data._db().execute("select name from portfolios order by name")]}
    if request.args.get("load"):
        b = data._db().execute("select body from portfolios where name=?", (request.args["load"],)).fetchone()
        if b: ctx["rows"] = json.loads(b[0])
    if request.method == "POST":
        ctx.update(rows=list(zip(f.getlist("t"), f.getlist("p"))), start=f["start"], end=f["end"], cache="cache" in f)
        try:
            w = parse(f)
            if f["act"] == "save":
                c = data._db(); c.execute("insert or replace into portfolios values(?,?)", (f["name"] or "unnamed", json.dumps([(t, str(round(v * 100, 4))) for t, v in w.items()]))); c.commit()
                ctx["saved"] = sorted(set(ctx["saved"]) | {f["name"] or "unnamed"})
            else:
                ctx["res"] = run(w, ctx["start"], ctx["end"], ctx["cache"])
                if f["act"] == "csv": return export(ctx["res"])
        except Exception as e:
            ctx["err"] = str(e)
    ctx["rows"] = ctx["rows"] + [("", "")] * max(0, 5 - len(ctx["rows"]))
    return render_template("index.html", **ctx)

def export(res):
    o = io.StringIO(); c = csv.writer(o)
    c.writerow(["metric", "portfolio", "benchmark"])
    for k, v in res["m"].items(): c.writerow([k, v, res["b"].get(k, "")])
    c.writerow([]); c.writerow(["test", "numbers", "result"]); c.writerows(res["tests"])
    c.writerow([]); c.writerow(["ticker", "weight", "beta", "alpha", "vol", "risk_share"])
    for h in res["hold"]: c.writerow(h.values())
    return Response(o.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=results.csv"})

if __name__ == "__main__":
    app.run(port=5000)
