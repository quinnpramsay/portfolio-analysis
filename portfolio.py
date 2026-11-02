import pandas as pd

def portfolio_returns(prices, weights):
    """Daily returns of a portfolio with target weights reset on the first trading day of each month.
    Weights drift between resets. prices: DataFrame of adjusted closes, no gaps."""
    r = prices.pct_change().dropna()
    w = pd.Series(weights).reindex(r.columns)
    out = []
    for _, m in r.groupby(r.index.to_period("M")):
        v = ((1 + m).cumprod() * w).sum(axis=1)       # value of 1 at start of month
        out.append(v / v.shift(1).fillna(1.0) - 1)
    return pd.concat(out)
