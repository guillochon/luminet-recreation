"""
Two black holes at the moment their horizons touch, inside a thin accretion disk,
drawn as a stippled ink image in the style of Luminet's 1979 picture.

Spacetime: two non-spinning holes on a circular orbit, written as a superposition
of two boosted Kerr-Schild metrics (as in Combi & Ressler 2024, arXiv:2403.13308),

    g = eta + H1 l1 l1 + H2 l2 l2,      H = 2m / r',

where r' is the distance from each hole in its own rest frame and l is its
Kerr-Schild null vector, boosted to the hole's orbital velocity. The holes are
placed at the separation where their coordinate horizons (r' = 2m) touch,
d = 2 m1 + 2 m2 = 2M, and held there: the inspiral during a light ray's transit
is ignored. The orbit is Keplerian, Omega = sqrt(M / d^3), so each equal-mass hole
moves at 0.35c. This is an approximation: superposed metrics are not exact
solutions where the two holes' fields are strong together.

Light: every pixel's ray is traced back from a distant camera through the
time-dependent metric, with the full geodesic (Hamiltonian) equations, so the
holes move while the light crosses them. The metric is unchanged by a time shift
combined with a rotation (it has a helical symmetry), so p_t + Omega p_phi is
conserved along each ray; p_t is taken from that instead of being integrated.

Disk: a thin circumbinary disk in the orbital plane, from r_in (default 6M, the
innermost stable orbit of the total mass) outward, orbiting the same way as the
binary, with the Page-Thorne brightness of a hole of the total mass and Luminet's
sharp inner edge. Its redshift is computed from each ray's own momentum and the
gas's 4-velocity: 1+z = (p.u)_emitter / (p.u)_camera.

Lengths are in units of the total mass M (G = c = 1).

    python merger.py                               # equal masses, side by side, Luminet's 80 degrees
    python merger.py --preset separated --out binary.png   # 12M apart: two separate shadows
    python merger.py --set phase=90                # one hole in front of the other
    python merger.py --set mass_ratio=0.5 --out merger_q05.png
    python merger.py --set separation=0            # one hole of mass M: reproduces blackhole.py
"""
import argparse
import dataclasses
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

import blackhole as bh
from blackhole import _p

MIN_MARGIN = 0.1          # rays reaching where det(g)/det(eta) < this count as fallen in (see lorentzian_margin)
SIGMA = -1.0             # Kerr-Schild null vector l = (1, SIGMA n): -1 is the outgoing form, regular on the past
                          # horizon, which rays traced backward in time cross smoothly


@dataclass
class MergerParams(bh.Params):
    separation: float = _p(None, "distance between the holes, in M; default (touching) is where their horizons touch, "
                                 "2 m1 + 2 m2 = 2M; 0 = a single hole of mass M")
    mass_ratio: float = _p(1.0, "m2 / m1, from 0.05 to 1")
    phase: float = _p(0.0, "orbital phase as the light passes the holes, in degrees: 0 = side by side with the heavier "
                           "hole on the right (receding), 90 = heavier hole in front")
    center_x: float = _p(0.0, "where the frame is centered on the sky, horizontally, in M (pans the view; 0 = on the binary)")
    center_y: float = _p(0.0, "where the frame is centered on the sky, vertically, in M (positive pans up)")
    camera_distance: float = _p(1000.0, "distance of the camera, in M; rays start here")
    step: float = _p(0.02, "integration step near the holes, as a fraction of the distance to the nearer hole")
    max_steps: int = _p(6000, "rays still orbiting after this many steps count as falling in")

    def resolved(self):
        p = super().resolved()
        d = 2.0 if self.separation is None else self.separation
        return dataclasses.replace(p, separation=d)

    def validate(self):
        super().validate()
        if self.spin != 0:
            raise ValueError("merger.py's holes do not spin; leave spin at 0")
        if not 0.05 <= self.mass_ratio <= 1:
            raise ValueError("mass_ratio must be in [0.05, 1]")
        if self.separation < 0:
            raise ValueError("separation must be >= 0")
        if self.separation > 0 and self.separation < 1.0:
            raise ValueError("separation below 1M puts the holes' centers inside each other's horizons")
        reach = max(self.separation * self.mass_ratio, self.separation) / (1 + self.mass_ratio) + 2 / (1 + self.mass_ratio)
        if self.r_in <= reach:
            raise ValueError(f"r_in must be outside the holes' orbits: more than {reach:.2f}M here "
                             f"(a circumbinary disk is cleared out to ~1.5-2x the separation)")


# Starting points: the stills' presets, plus binary-specific ones.
PRESETS = {**bh.PRESETS,
           # Holes far enough apart that each casts its own shadow; the disk is cleared
           # out to 1.5x the separation, as the binary's tides clear a circumbinary disk.
           'separated': {'separation': 12.0, 'r_in': 18.0, 'view': 28.0}}


# --- The binary --------------------------------------------------------------------
class Binary:
    """Two holes on a circular orbit in the z = 0 plane, counterclockwise seen from +z."""
    def __init__(self, d, q):
        self.m = (1 / (1 + q), q / (1 + q))                 # m1 >= m2, total 1
        self.d = d
        self.omega = np.sqrt(1 / d**3) if d > 0 else 0.0
        self.radius = (d * self.m[1], -d * self.m[0])       # signed distance of each hole from the center of mass
        self.psi0 = 0.0                                     # orbital angle at t = 0 (set by the caller)

    def hole(self, i, t):
        psi = self.psi0 + self.omega * t
        c, s = np.cos(psi), np.sin(psi)
        R = self.radius[i]
        X = np.stack([R * c, R * s, np.zeros_like(c)])
        V = np.stack([-R * self.omega * s, R * self.omega * c, np.zeros_like(c)])
        return X, V


def hole_terms(x, X, V, m, derivs=True):
    """Kerr-Schild pieces of one moving hole at points x (3, N): H, l_mu = (l_t, l_s)
    and, if derivs, their spatial derivatives, packed as what the geodesic equation needs.

    In the hole's rest frame, s' = A s with A = 1 + kappa v v^T (kappa = gamma^2/(gamma+1)),
    r' = |s'|, n = s'/r', H = 2m/r', l' = (1, SIGMA n); boosting l' back gives
        l_t = gamma (1 - SIGMA v.n),    l_s = SIGMA A n - gamma v.
    """
    s = x - X
    v2 = np.sum(V * V, 0)
    g = 1 / np.sqrt(1 - v2)
    kap = g * g / (g + 1)
    sv = np.sum(s * V, 0)
    sp = s + kap * sv * V
    r = np.sqrt(np.sum(sp * sp, 0))
    n = sp / r
    nv = np.sum(n * V, 0)
    w = n + kap * nv * V                                    # A n
    H = 2 * m / r
    lt = g * (1 - SIGMA * nv)
    ls = SIGMA * w - g * V
    out = dict(r=r, H=H, lt=lt, ls=ls)
    if derivs:
        out.update(g=g, V=V, nv=nv, w=w)
    return out


def lorentzian_margin(T):
    """det(g) / det(eta) = 1 - H1 H2 (l1.l2)^2. Where it reaches 0 the superposed metric
    stops being a spacetime metric. That happens only near the neck between two
    touching horizons (up to ~0.8M from the touching point, for equal masses), a
    region inside the binary's shadow; rays that get there are counted as fallen in."""
    a, b = T[0]['H'], T[1]['H']
    lm = -T[0]['lt'] * T[1]['lt'] + np.sum(T[0]['ls'] * T[1]['ls'], 0)
    return 1 - a * b * lm * lm


def raise_index(p_t, p_s, T):
    """k = g^-1 p for g = eta + H1 l1 l1 + H2 l2 l2 (Woodbury identity; each l is null under eta)."""
    a, b = T[0]['H'], T[1]['H']
    l, mm = T[0], T[1]
    lp = -l['lt'] * p_t + np.sum(l['ls'] * p_s, 0)
    mp = -mm['lt'] * p_t + np.sum(mm['ls'] * p_s, 0)
    lm = -l['lt'] * mm['lt'] + np.sum(l['ls'] * mm['ls'], 0)
    den = 1 - a * b * lm * lm
    w1 = (a * lp - a * b * lm * mp) / den
    w2 = (b * mp - a * b * lm * lp) / den
    k_t = -p_t + w1 * l['lt'] + w2 * mm['lt']               # eta raises t with a minus sign
    k_s = p_s - w1 * l['ls'] - w2 * mm['ls']
    return k_t, k_s


def force(k_t, k_s, T):
    """dp_j/dlambda = 1/2 k^a k^b d_j g_ab = sum over holes of
    1/2 d_j H (k.l)^2 + H (k.l) (k^a d_j l_a)."""
    F = 0.0
    for h in T:
        r, H, g, V, nv, w = h['r'], h['H'], h['g'], h['V'], h['nv'], h['w']
        kl = k_t * h['lt'] + np.sum(k_s * h['ls'], 0)
        kv = np.sum(k_s * V, 0)
        kw = np.sum(k_s * w, 0)
        dH = -H * w / r
        # k^a d_j l_a, with d_j n_k = (A_kj - n_k w_j)/r' and A A = 1 + gamma^2 v v^T
        kdl = (-g * SIGMA * k_t * (g * V - nv * w) + SIGMA * (k_s + g * g * kv * V - kw * w)) / r
        F = F + 0.5 * dH * kl * kl + H * kl * kdl
    return F


def metric_terms(binary, t, x, derivs=True):
    return [hole_terms(x, *binary.hole(i, t), binary.m[i], derivs) for i in (0, 1)]


# --- Rays ------------------------------------------------------------------------
def camera_frame(inclination):
    """Unit vector toward the camera and the screen's right and up directions. The
    camera looks from the -y side, so the disk's near side is at the bottom of the
    screen and its approaching (counterclockwise-moving) side on the left."""
    i = np.radians(min(inclination, 89.999))
    n = np.array([0.0, -np.sin(i), np.cos(i)])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, np.cos(i), np.sin(i)])
    return n, right, up


def initial_rays(p, binary, X, Y):
    """Positions, times and null momenta of rays leaving the camera's screen at (X, Y)."""
    n, right, up = camera_frame(p.inclination)
    D = p.camera_distance
    x = D * n[:, None] + X[None] * right[:, None] + Y[None] * up[:, None]
    t = np.full(X.shape, D)
    p_s = np.repeat(n[:, None], X.size, 1)                  # the photon travels toward the camera
    T = metric_terms(binary, t, x, derivs=False)
    # solve g^ab p_a p_b = 0 for p_t, with k^t > 0 (future pointing)
    At, _ = raise_index(np.ones_like(t), 0 * p_s, T)        # g^tt
    Bt, Bs = raise_index(np.zeros_like(t), p_s, T)          # g^tj p_j and g^jk p_k
    Cq = np.sum(p_s * Bs, 0)
    disc = np.sqrt(Bt * Bt - At * Cq)
    roots = np.stack([(-Bt + disc) / At, (-Bt - disc) / At])
    kt = At[None] * roots + Bt[None]
    p_t = np.where(kt[0] > 0, roots[0], roots[1])
    E_cam = -p_t                                            # energy at infinity: the camera is far away, as in Luminet
    return x, t, p_s, p_t, E_cam


def trace_rays(p, binary_args, X, Y):
    """Trace a batch of rays backward from the camera. Returns, per ray, the disk
    radius where it first meets the disk (NaN if never), how many times it had
    crossed the disk plane before that, the redshift 1+z of the light, and which
    hole swallowed it (0 = heavier, 1 = lighter, 2 = still orbiting at max_steps, -1 = none)."""
    binary = Binary(*binary_args[:2]); binary.psi0 = binary_args[2]
    N = X.size
    x, t, p_s, p_t, E_cam = initial_rays(p, binary, X, Y)
    C = p_t + binary.omega * (x[0] * p_s[1] - x[1] * p_s[0])   # conserved: p_t + Omega p_phi
    rho_hit = np.full(N, np.nan); n_cross = np.zeros(N, int); opz = np.full(N, np.nan)
    captor = np.full(N, -1, np.int8)                        # which hole swallowed the ray (-1: none)
    idx = np.arange(N)
    crossings = np.zeros(N, int)
    m = binary.m
    D = p.camera_distance

    def rhs(t, x, ps, C):
        pt = C - binary.omega * (x[0] * ps[1] - x[1] * ps[0])
        T = metric_terms(binary, t, x)
        kt, ks = raise_index(pt, ps, T)
        return kt, ks, force(kt, ks, T), T

    for step in range(p.max_steps):
        kt1, ks1, F1, T = rhs(t, x, p_s, C)
        r_near = np.minimum(T[0]['r'] / (2 * m[0]), T[1]['r'] / (2 * m[1]))
        r_hole = np.minimum(T[0]['r'], T[1]['r'])
        done = (r_near < 0.5) | (lorentzian_margin(T) < MIN_MARGIN) | ~np.isfinite(kt1)   # fallen in
        out = (np.sum(x * x, 0) > (1.05 * D)**2)            # escaped past the camera's distance
        keep = ~(done | out)
        if not keep.all():
            captor[idx[done]] = np.where(T[0]['r'][done] / m[0] <= T[1]['r'][done] / m[1], 0, 1)
            idx, x, t, p_s, C, crossings = idx[keep], x[:, keep], t[keep], p_s[:, keep], C[keep], crossings[keep]
            kt1, ks1, F1 = kt1[keep], ks1[:, keep], F1[:, keep]
            r_hole = r_hole[keep]
            if len(idx) == 0:
                break
        # step size ~ distance to the nearer hole, coarser far away; the light's coordinate speed
        # grows near the holes, so scale by it to keep each step a small fraction of the distance
        frac = p.step * (1 + np.clip((r_hole - 30) / 30, 0, 4))
        speed = np.maximum(np.sqrt(np.sum(ks1 * ks1, 0)), np.abs(kt1))
        h = -frac * r_hole / speed                          # negative: backward along the ray
        kt2, ks2, F2, _ = rhs(t + 0.5*h*kt1, x + 0.5*h*ks1, p_s + 0.5*h*F1, C)
        kt3, ks3, F3, _ = rhs(t + 0.5*h*kt2, x + 0.5*h*ks2, p_s + 0.5*h*F2, C)
        kt4, ks4, F4, _ = rhs(t + h*kt3, x + h*ks3, p_s + h*F3, C)
        t_new = t + h/6 * (kt1 + 2*kt2 + 2*kt3 + kt4)
        x_new = x + h/6 * (ks1 + 2*ks2 + 2*ks3 + ks4)
        p_new = p_s + h/6 * (F1 + 2*F2 + 2*F3 + F4)
        crossed = (x[2] > 0) != (x_new[2] > 0)
        if crossed.any():
            i = np.nonzero(crossed)[0]
            f = x[2, i] / (x[2, i] - x_new[2, i])
            xc = x[:, i] + f * (x_new[:, i] - x[:, i])
            rho = np.hypot(xc[0], xc[1])
            hit = (rho >= p.r_in) & (rho <= p.r_out)
            if hit.any():
                j = i[hit]
                tc = t[j] + f[hit] * (t_new[j] - t[j])
                pc = p_s[:, j] + f[hit] * (p_new[:, j] - p_s[:, j])
                xc = xc[:, hit]
                rho_hit[idx[j]] = rho[hit]
                n_cross[idx[j]] = crossings[j]
                opz[idx[j]] = emitted_energy(binary, tc, xc, pc, C[j]) / E_cam[idx[j]]
            crossings[i] += 1
            # rays that met the disk stop there (it is opaque)
            if hit.any():
                keep = np.ones(len(idx), bool); keep[i[hit]] = False
                idx, x_new, t_new, p_new, C, crossings = (idx[keep], x_new[:, keep], t_new[keep], p_new[:, keep],
                                                          C[keep], crossings[keep])
        x, t, p_s = x_new, t_new, p_new
        if len(idx) == 0:
            break
    captor[idx] = 2                                         # still orbiting at the step limit
    return rho_hit, n_cross, opz, captor


def emitted_energy(binary, t, x, p_s, C):
    """The light's energy -p.u measured by the disk gas at x; divided by the energy
    the camera measures, it gives 1+z. The gas orbits at the
    Keplerian rate of the total mass, Omega_d = rho^-3/2, which is exact for a
    single Schwarzschild hole in Kerr-Schild coordinates."""
    p_t = C - binary.omega * (x[0] * p_s[1] - x[1] * p_s[0])
    rho = np.hypot(x[0], x[1])
    Om = rho**-1.5
    K_t, K_s = np.ones_like(rho), np.stack([-Om * x[1], Om * x[0], 0 * rho])
    gKK = -1 + np.sum(K_s * K_s, 0)
    for h in metric_terms(binary, t, x, derivs=False):
        gKK = gKK + h['H'] * (h['lt'] * K_t + np.sum(h['ls'] * K_s, 0))**2
    ut = 1 / np.sqrt(-gKK)
    pu = ut * (p_t + Om * (x[0] * p_s[1] - x[1] * p_s[0]))   # p_mu u^mu, negative
    return -pu


# --- The image -----------------------------------------------------------------
def phase_offset(p, binary):
    """Orbital angle at t = 0 that puts the holes at p.phase when light passing
    near them crosses the plane through the center facing the camera."""
    binary.psi0 = 0.0
    n, _, _ = camera_frame(p.inclination)
    x, t, p_s, p_t, _ = initial_rays(p, binary, np.array([3.0 * max(binary.d, 2)]), np.array([0.0]))
    # light near the center arrives ~D + Shapiro delay after leaving: measure it on one straight-ish ray
    C = p_t + binary.omega * (x[0] * p_s[1] - x[1] * p_s[0])
    for _ in range(100000):
        pt = C - binary.omega * (x[0] * p_s[1] - x[1] * p_s[0])
        T = metric_terms(binary, t, x)
        kt, ks = raise_index(pt, p_s, T)
        F = force(kt, ks, T)
        r = min(T[0]['r'][0], T[1]['r'][0])
        h = -0.01 * max(r, 1.0)
        x_new = x + h * ks; t_new = t + h * kt; p_s = p_s + h * F
        if np.dot(n, x_new[:, 0]) <= 0:
            t_ref = float(t_new[0]); break
        x, t = x_new, t_new
    # psi is the heavier hole's angle in the orbital plane from +x (screen right); the camera is
    # toward -y, so phase 0 (psi = 0) is side by side and phase 90 (psi = -90 deg) puts it in front
    psi = np.radians(-p.phase)
    return psi - binary.omega * t_ref


def binary_image(p, verbose=True, workers=None):
    """Trace every image cell and return the dict blackhole.stipple expects."""
    VIEW, VIEW_H = p.view, p.view_h
    W = p.grid_w; H = int(round(W / p.aspect))
    xs = np.linspace(-VIEW, VIEW, W); ys = np.linspace(-VIEW_H, VIEW_H, H)
    X, Y = np.meshgrid(xs, ys)
    binary = Binary(p.separation, p.mass_ratio)
    psi0 = phase_offset(p, binary) if binary.d > 0 else 0.0
    args = (binary.d, p.mass_ratio, psi0)
    Xf, Yf = X.ravel() + p.center_x, Y.ravel() + p.center_y          # where on the sky each cell looks
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    chunks = np.array_split(np.random.default_rng(0).permutation(len(Xf)), workers * 6)   # mixed, for even loads
    t0 = time.time()
    if verbose:
        print(f"  tracing {len(Xf)} rays past the binary (separation {binary.d:g}M, masses "
              f"{binary.m[0]:.3f}/{binary.m[1]:.3f}) on {workers} processes...", flush=True)
    rho = np.full(len(Xf), np.nan); ncr = np.zeros(len(Xf), int); opz = np.full(len(Xf), np.nan)
    cap = np.full(len(Xf), -1, np.int8)
    if workers == 1:
        parts = [trace_rays(p, args, Xf[c], Yf[c]) for c in chunks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(trace_rays, [p] * len(chunks), [args] * len(chunks),
                                [Xf[c] for c in chunks], [Yf[c] for c in chunks]))
    for c, (r_, n_, z_, k_) in zip(chunks, parts):
        rho[c], ncr[c], opz[c], cap[c] = r_, n_, z_, k_
    if verbose: print(f"  traced in {time.time() - t0:.0f} s", flush=True)
    rho, ncr, opz = rho.reshape(X.shape), ncr.reshape(X.shape), opz.reshape(X.shape)
    hit = ~np.isnan(rho)
    with np.errstate(invalid='ignore'):
        F = disk_flux(p, np.where(hit, rho, 10.0)) / opz**4
    flux = np.where(hit, F, 0.0)
    B = np.hypot(X, Y)
    ALPHA = np.arctan2(X, -Y)
    PRIM = hit & (ncr == 0)
    SEC = hit & (ncr > 0)
    return dict(xs=xs, ys=ys, W=W, H=H, B=B, ALPHA=ALPHA, flux=flux, RMAP=np.where(hit, rho, np.nan),
                PRIM=PRIM, SEC=SEC, opz=opz, captor=cap.reshape(X.shape))


def disk_flux(p, r):
    """Emitted Page-Thorne flux of a disk around the total mass, with Luminet's sharp inner edge."""
    if p.sharp_inner_edge:
        fade = 1 - np.exp(-np.maximum(r - p.r_in, 0) / p.edge_softness)
        return fade * bh.page_thorne(np.maximum(r, p.r_peak))
    return bh.page_thorne(r)


def make_drawing(p, verbose=True, img=None):
    p = p.resolved()
    p.validate()
    img = img or binary_image(p, verbose)
    dots, rng = bh.stipple(p, img, verbose)
    size = p.dot_size * rng.lognormal(0, p.dot_size_jitter, len(dots))
    ang = rng.uniform(0, 2 * np.pi, len(dots)); off = rng.uniform(p.lobe_offset_min, p.lobe_offset_max, len(dots))
    pt_to_M = 2 * p.view / (bh.FIG_W * 72)
    lobe = dots + np.column_stack([np.cos(ang), np.sin(ang)]) * (off * size * pt_to_M)[:, None]
    if p.approaching_side == 'right':
        dots = dots * [-1, 1]; lobe = lobe * [-1, 1]
    return bh.Drawing(p, dots, size, lobe)


# --- Command line ------------------------------------------------------------------
def params_from_args(args):
    if args.list_params:
        for f in dataclasses.fields(MergerParams):
            print(f"  {f.name:22s} {f.default!r:12}  {f.metadata['help']}")
        raise SystemExit
    values = dict(PRESETS[args.preset])
    if args.params:
        with open(args.params) as fh:
            values.update(json.load(fh))
    for item in args.set:
        name, _, value = item.partition('=')
        values[name.strip()] = value.strip()
    kinds = {f.name: f.type for f in dataclasses.fields(MergerParams)}
    def conv(k, v):
        if k not in kinds:
            raise SystemExit(f"unknown parameter {k!r}; see --list-params")
        if k == 'separation' and (v is None or str(v).lower() in ('touching', 'auto', 'none', '')):
            return None
        if k in ('separation', 'mass_ratio', 'phase', 'camera_distance', 'step', 'center_x', 'center_y'):
            return float(v)
        if k == 'max_steps':
            return int(v)
        return bh._convert(k, v)
    return MergerParams(**{k: conv(k, v) for k, v in values.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='merger.png', help='output PNG; the parameters are saved next to it as .json')
    ap.add_argument('--res-scale', type=float, default=2, help='resolution multiplier (1 = 2800 px wide)')
    bh.add_param_args(ap, PRESETS)
    args = ap.parse_args()
    p = params_from_args(args)
    d = make_drawing(p)
    bh.render_still(d, args.out, args.res_scale)
    with open(args.out.rsplit('.', 1)[0] + '.json', 'w') as fh:
        json.dump(dataclasses.asdict(d.p), fh, indent=2)
    print(f"{len(d.dots)} dots -> {args.out}")


if __name__ == '__main__':
    main()
