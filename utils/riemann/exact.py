"""Exact 1D Euler Riemann solver (Toro, Riemann Solvers, ch. 4).

The kernel never calls this. Tests and examples/hd_eval use it as the
yardstick. ``exact_hi`` wraps number-density / cgs ICs into the same
code units as ``xmf.shock_tube()`` (UNIT_DENSITY, UNIT_VELOCITY).
"""
from __future__ import annotations

import math

GAMMA = 5.0 / 3.0
MASS_H = 1.00794
AMU = 1.66053886e-24
UNIT_DENSITY = 1.0e-12
UNIT_VELOCITY = 1.0e6


def n_to_rho(n: float) -> float:
    return n * MASS_H * AMU / UNIT_DENSITY


def rho_to_n(rho: float) -> float:
    return rho * UNIT_DENSITY / (MASS_H * AMU)


def _sound(rho: float, p: float, gamma: float) -> float:
    return math.sqrt(gamma * p / rho)


def _star_flux(p: float, rho_k: float, p_k: float, a_k: float, gamma: float) -> tuple[float, float]:
    """Pressure function f_K(p) and its derivative (Toro 4.6–4.7)."""
    gp, gm = gamma + 1.0, gamma - 1.0
    if p > p_k:
        a = 2.0 / (gp * rho_k)
        b = gm / gp * p_k
        den = math.sqrt(a / (p + b))
        f = (p - p_k) * den
        df = den * (1.0 - 0.5 * (p - p_k) / (p + b))
        return f, df
    pr = p / p_k
    f = 2.0 / gm * a_k * (pr ** (gm / (2.0 * gamma)) - 1.0)
    df = 1.0 / (rho_k * a_k) * pr ** (-(gp) / (2.0 * gamma))
    return f, df


def p_star(wl: tuple[float, float, float], wr: tuple[float, float, float], gamma: float) -> tuple[float, float]:
    """Iterate p* and return (p*, v*). States are (rho, v, p) in code units."""
    rho_l, v_l, p_l = wl
    rho_r, v_r, p_r = wr
    a_l, a_r = _sound(rho_l, p_l, gamma), _sound(rho_r, p_r, gamma)
    p = 0.5 * (p_l + p_r)
    for _ in range(40):
        fl, dfl = _star_flux(p, rho_l, p_l, a_l, gamma)
        fr, dfr = _star_flux(p, rho_r, p_r, a_r, gamma)
        dp = (fl + fr + (v_r - v_l)) / (dfl + dfr)
        p = max(p - dp, 1.0e-12 * min(p_l, p_r))
        if abs(dp) < 1.0e-10 * p:
            break
    v = 0.5 * (v_l + v_r + fr - fl)
    return p, v


def sample(
    xi: float,
    wl: tuple[float, float, float],
    wr: tuple[float, float, float],
    p_s: float,
    v_s: float,
    gamma: float,
) -> tuple[float, float, float]:
    """Self-similar sample at xi = (x - x0) / t. Returns (rho, v, p)."""
    rho_l, v_l, p_l = wl
    rho_r, v_r, p_r = wr
    a_l, a_r = _sound(rho_l, p_l, gamma), _sound(rho_r, p_r, gamma)
    gp, gm = gamma + 1.0, gamma - 1.0

    if xi <= v_s:
        if p_s > p_l:
            s = v_l - a_l * math.sqrt(gp / (2.0 * gamma) * p_s / p_l + gm / (2.0 * gamma))
            if xi <= s:
                return rho_l, v_l, p_l
            rho = rho_l * (p_s / p_l * gp / gm + 1.0) / (gp / gm + p_s / p_l)
            return rho, v_s, p_s
        a_sl = a_l * (p_s / p_l) ** (gm / (2.0 * gamma))
        sh, st = v_l - a_l, v_s - a_sl
        if xi <= sh:
            return rho_l, v_l, p_l
        if xi >= st:
            rho = rho_l * (p_s / p_l) ** (1.0 / gamma)
            return rho, v_s, p_s
        v = 2.0 / gp * (a_l + 0.5 * gm * v_l + xi)
        a = 2.0 / gp * (a_l + 0.5 * gm * (v_l - xi))
        rho = rho_l * (a / a_l) ** (2.0 / gm)
        p = p_l * (a / a_l) ** (2.0 * gamma / gm)
        return rho, v, p

    if p_s > p_r:
        s = v_r + a_r * math.sqrt(gp / (2.0 * gamma) * p_s / p_r + gm / (2.0 * gamma))
        if xi >= s:
            return rho_r, v_r, p_r
        rho = rho_r * (p_s / p_r * gp / gm + 1.0) / (gp / gm + p_s / p_r)
        return rho, v_s, p_s
    a_sr = a_r * (p_s / p_r) ** (gm / (2.0 * gamma))
    sh, st = v_r + a_r, v_s + a_sr
    if xi >= sh:
        return rho_r, v_r, p_r
    if xi <= st:
        rho = rho_r * (p_s / p_r) ** (1.0 / gamma)
        return rho, v_s, p_s
    v = 2.0 / gp * (-a_r + 0.5 * gm * v_r + xi)
    a = 2.0 / gp * (a_r - 0.5 * gm * (v_r - xi))
    rho = rho_r * (a / a_r) ** (2.0 / gm)
    p = p_r * (a / a_r) ** (2.0 * gamma / gm)
    return rho, v, p


def exact_hi(
    x,
    t: float,
    x0: float,
    n_l: float,
    n_r: float,
    v_l: float,
    v_r: float,
    p_l: float,
    p_r: float,
    gamma: float = GAMMA,
):
    """Exact HI profiles. x, t are code units; n, v, p are cgs."""
    import numpy as np

    wl = (n_to_rho(n_l), v_l / UNIT_VELOCITY, p_l)
    wr = (n_to_rho(n_r), v_r / UNIT_VELOCITY, p_r)
    ps, vs = p_star(wl, wr, gamma)
    n = np.empty_like(x, dtype=float)
    v = np.empty_like(x, dtype=float)
    p = np.empty_like(x, dtype=float)
    if t <= 0.0:
        left = x < x0
        n[left], v[left], p[left] = n_l, v_l, p_l
        n[~left], v[~left], p[~left] = n_r, v_r, p_r
        return n, v, p
    for i, xi in enumerate((x - x0) / t):
        rho, vv, pp = sample(float(xi), wl, wr, ps, vs, gamma)
        n[i] = rho_to_n(rho)
        v[i] = vv * UNIT_VELOCITY
        p[i] = pp
    return n, v, p
