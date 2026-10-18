def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def decimal_to_american(d: float) -> int:
    if d >= 2:
        return round((d - 1) * 100)
    return round(-100 / (d - 1))


def prob_to_american(p: float) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return decimal_to_american(1 / p)


def devig(*decimal_prices: float) -> list[float]:
    """Remove the margin proportionally (multiplicative method)."""
    raw = [1 / d for d in decimal_prices]
    s = sum(raw)
    return [x / s for x in raw]


def map_prob_from_series(p_series: float, best_of: int) -> float:
    """Per-map win prob implied by a series win prob, assuming i.i.d. maps."""
    if best_of <= 1:
        return p_series
    need = best_of // 2 + 1

    def series(p):
        from math import comb
        return sum(comb(need - 1 + k, k) * p**need * (1 - p) ** k for k in range(need))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if series(mid) < p_series:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
