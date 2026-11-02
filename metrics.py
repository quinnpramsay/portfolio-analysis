import numpy as np, pandas as pd
D = 252

def ols_hac(y, x, lags=5):
    """OLS of y on [1,x] with Newey-West (Bartlett) standard errors. Returns alpha, beta, se_alpha, se_beta."""
    X = np.column_stack([np.ones(len(x)), x]); n = len(y)
    b = np.linalg.solve(X.T @ X, X.T @ y); u = y - X @ b
    Xu = X * u[:, None]; S = Xu.T @ Xu
    for l in range(1, lags + 1):
        G = Xu[l:].T @ Xu[:-l]; S += (1 - l / (lags + 1)) * (G + G.T)
    A = np.linalg.inv(X.T @ X); V = A @ S @ A
    return b[0], b[1], np.sqrt(V[0, 0]), np.sqrt(V[1, 1])

def cagr(r): return (1 + r).prod() ** (D / len(r)) - 1
def vol(r): return r.std(ddof=1) * np.sqrt(D)

def max_drawdown(r):
    w = (1 + r).cumprod(); dd = w / w.cummax() - 1
    trough = dd.idxmin(); peak = w.loc[:trough].idxmax()
    return dd.min(), peak, trough

def capture(rp, rm):
    up, dn = rm > 0, rm < 0
    return rp[up].mean() / rm[up].mean() * 100, rp[dn].mean() / rm[dn].mean() * 100

def sharpe(r, rf): return (r - rf).mean() / (r - rf).std(ddof=1) * np.sqrt(D)

def jensen(rp, rm, rf):
    a, b, sa, sb = ols_hac((rp - rf).values, (rm - rf).values)
    return {"alpha": a * D, "alpha_lo": (a - 1.96 * sa) * D, "alpha_hi": (a + 1.96 * sa) * D, "t": a / sa, "beta": b}

def analyze(rp, rm, rf):
    j = jensen(rp, rm, rf)
    uc, dc = capture(rp, rm)
    mdd, pk, tr = max_drawdown(rp); bmdd = max_drawdown(rm)[0]
    mix = j["beta"] * rm + (1 - j["beta"]) * rf          # beta-matched mix of benchmark and cash
    mix_c, mix_d = cagr(mix), max_drawdown(mix)[0]
    m = {**j, "corr": rp.corr(rm), "cagr": cagr(rp), "vol": vol(rp), "mdd": mdd, "mdd_peak": pk, "mdd_trough": tr,
         "down_dev": np.sqrt((np.minimum(rp, 0) ** 2).mean()) * np.sqrt(D), "up_cap": uc, "down_cap": dc,
         "sharpe": sharpe(rp, rf), "active": cagr(rp) - cagr(rm)}
    b = {"cagr": cagr(rm), "vol": vol(rm), "mdd": bmdd, "sharpe": sharpe(rm, rf),
         "down_dev": np.sqrt((np.minimum(rm, 0) ** 2).mean()) * np.sqrt(D)}
    verdict = "efficient" if j["t"] >= 1.96 else "inefficient" if j["t"] <= -1.96 else "inconclusive"
    better = int(m["cagr"] > mix_c) + int(m["mdd"] > mix_d)
    tests = [
        ("Alpha t-stat (95%): headline verdict", f"t = {j['t']:.2f}", verdict),
        ("Downside capture at or below beta x 100", f"{dc:.1f} vs {j['beta'] * 100:.1f}", "pass" if dc <= j["beta"] * 100 else "fail"),
        ("Upside capture at or above downside capture", f"{uc:.1f} vs {dc:.1f}", "pass" if uc >= dc else "fail"),
        ("Beats beta-matched benchmark/cash mix (return and drawdown)",
         f"return {m['cagr']:.2%} vs {mix_c:.2%}; drawdown {mdd:.2%} vs {mix_d:.2%}", ["fail", "mixed", "pass"][better]),
        ("Sharpe at or above benchmark Sharpe", f"{m['sharpe']:.2f} vs {b['sharpe']:.2f}", "pass" if m["sharpe"] >= b["sharpe"] else "fail"),
    ]
    return m, b, verdict, tests

def holdings(returns, rp, rm, rf, weights):
    rows, var = [], rp.var()
    for t, w in weights.items():
        r = returns[t]; j = jensen(r, rm, rf)
        rows.append({"ticker": t, "weight": w, "beta": j["beta"], "alpha": j["alpha"], "vol": vol(r),
                     "risk_share": w * r.cov(rp) / var})
    return rows
