import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, pandas as pd, statsmodels.api as sm
import metrics
from portfolio import portfolio_returns

def rand(n=1500, seed=1):
    g = np.random.default_rng(seed); i = pd.bdate_range("2018-01-01", periods=n)
    return pd.Series(g.normal(0.0004, 0.01, n), i), pd.Series(0.00008, i), i, g

def test_benchmark_itself():
    rm, rf, *_ = rand()
    m, b, v, _ = metrics.analyze(rm, rm, rf)
    assert abs(m["beta"] - 1) < 1e-9 and abs(m["alpha"]) < 1e-9 and abs(m["corr"] - 1) < 1e-9
    assert abs(m["up_cap"] - 100) < 1e-9 and abs(m["down_cap"] - 100) < 1e-9

def test_known_beta_and_alpha():
    rm, rf, *_ = rand()
    j = metrics.jensen(rf + 0.0002 + 1.5 * (rm - rf), rm, rf)      # exact beta 1.5, daily alpha 0.0002
    assert abs(j["beta"] - 1.5) < 1e-9 and abs(j["alpha"] - 0.0002 * 252) < 1e-9

def test_regression_matches_statsmodels():
    rm, rf, i, g = rand(); rp = 0.0001 + 0.8 * rm + pd.Series(g.normal(0, 0.005, len(rm)), i)
    a, b, sa, sb = metrics.ols_hac((rp - rf).values, (rm - rf).values)
    r = sm.OLS((rp - rf).values, sm.add_constant((rm - rf).values)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    assert np.allclose([a, b], r.params) and np.allclose([sa, sb], r.bse)

def test_drawdown_by_hand():
    r = pd.Series([0.10, -0.20, 0.05, -0.10], pd.bdate_range("2020-01-01", periods=4))
    d, pk, tr = metrics.max_drawdown(r)                  # peak 1.10, trough 1.10*.8*1.05*.9 = 0.9702
    assert abs(d - (1.1 * 0.8 * 1.05 * 0.9 / 1.1 - 1)) < 1e-12 and tr == r.index[3] and pk == r.index[0]

def test_cagr_and_vol_by_hand():
    assert abs(metrics.cagr(pd.Series([0.01] * 252)) - (1.01 ** 252 - 1)) < 1e-12
    r = pd.Series([0.01, -0.01] * 126); assert abs(metrics.vol(r) - r.std(ddof=1) * np.sqrt(252)) < 1e-12

def test_monthly_rebalance_by_hand():
    i = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    px = pd.DataFrame({"A": [100, 110, 110, 110], "B": [100, 100, 100, 120]}, index=i)
    r = portfolio_returns(px, {"A": 0.5, "B": 0.5})      # Jan 31: 5%. Feb 3: 0. Feb 4: half in B at +20% = 10%
    assert np.allclose(r.values, [0.05, 0.0, 0.10])
