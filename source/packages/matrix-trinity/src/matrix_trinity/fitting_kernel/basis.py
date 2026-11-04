from __future__ import annotations

import math


def legendre_p(n, x):
    if n == 0:
        return 1.0
    if n == 1:
        return x
    p0, p1 = 1.0, x
    for k in range(2, n + 1):
        pk = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        p0, p1 = p1, pk
    return p1


def assoc_legendre_p(n, m, x):
    if m < 0 or n < m:
        raise ValueError("Invalid n,m for associated Legendre")
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt(max(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if n == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if n == m + 1:
        return pmmp1
    pnm2 = pmm
    pnm1 = pmmp1
    for nn in range(m + 2, n + 1):
        pn = ((2 * nn - 1) * x * pnm1 - (nn + m - 1) * pnm2) / (nn - m)
        pnm2, pnm1 = pnm1, pn
    return pnm1


def legendre_p_deriv(n, x):
    if n == 0:
        return 0.0
    if abs(x) >= 1.0:
        x = math.copysign(0.999999, x)
    pn = legendre_p(n, x)
    pn1 = legendre_p(n - 1, x)
    return (n / (x * x - 1.0)) * (x * pn - pn1)


def assoc_legendre_p_deriv(n, m, x):
    if abs(x) >= 1.0:
        x = math.copysign(0.999999, x)
    pn = assoc_legendre_p(n, m, x)
    pn1 = assoc_legendre_p(n - 1, m, x) if n > m else 0.0
    return (1.0 / (x * x - 1.0)) * (n * x * pn - (n + m) * pn1)


def basis_value(mode, x, exp, params):
    if exp == 0:
        return 1.0
    if mode == "poly":
        return x**exp
    if mode == "rational":
        s = params.get("shift", 0.0)
        f = x / (x + s)
        return f**exp
    if mode == "rational2":
        s = params.get("shift", 0.0)
        f = x / (x + 2.0 * s)
        return f**exp
    if mode == "trig":
        return math.sin(exp * x)
    if mode == "trig_cos":
        return math.cos(exp * x)
    if mode == "legendre":
        m = params.get("m", 0)
        return assoc_legendre_p(exp, m, x) if m > 0 else legendre_p(exp, x)
    if mode == "legendre_cos":
        m = params.get("m", 0)
        cx = math.cos(x)
        return assoc_legendre_p(exp, m, cx) if m > 0 else legendre_p(exp, cx)
    if mode == "exp":
        a = params.get("a", 1.0)
        x0 = params.get("x0", 0.0)
        f = math.exp(-a * (x - x0))
        return f**exp
    if mode == "morse":
        a = params.get("a", 1.0)
        x0 = params.get("x0", 0.0)
        f = 1.0 - math.exp(-a * (x - x0))
        return f**exp
    raise ValueError(f"Unknown basis mode: {mode}")


def basis_derivative(mode, x, exp, params):
    if exp == 0:
        return 0.0
    if mode == "poly":
        return exp * (x ** (exp - 1))
    if mode == "rational":
        s = params.get("shift", 0.0)
        denom = x + s
        if denom == 0.0:
            return 0.0
        f = x / denom
        df = s / (denom * denom)
        return exp * (f ** (exp - 1)) * df
    if mode == "rational2":
        s = params.get("shift", 0.0)
        denom = x + 2.0 * s
        if denom == 0.0:
            return 0.0
        f = x / denom
        df = (2.0 * s) / (denom * denom)
        return exp * (f ** (exp - 1)) * df
    if mode == "trig":
        return exp * math.cos(exp * x)
    if mode == "trig_cos":
        return -exp * math.sin(exp * x)
    if mode == "legendre":
        m = params.get("m", 0)
        return assoc_legendre_p_deriv(exp, m, x) if m > 0 else legendre_p_deriv(exp, x)
    if mode == "legendre_cos":
        m = params.get("m", 0)
        cx = math.cos(x)
        d = assoc_legendre_p_deriv(exp, m, cx) if m > 0 else legendre_p_deriv(exp, cx)
        return -math.sin(x) * d
    if mode == "exp":
        a = params.get("a", 1.0)
        x0 = params.get("x0", 0.0)
        f = math.exp(-a * (x - x0))
        df = -a * f
        return exp * (f ** (exp - 1)) * df
    if mode == "morse":
        a = params.get("a", 1.0)
        x0 = params.get("x0", 0.0)
        f = 1.0 - math.exp(-a * (x - x0))
        df = a * math.exp(-a * (x - x0))
        return exp * (f ** (exp - 1)) * df
    raise ValueError(f"Unknown basis mode: {mode}")
