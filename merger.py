"""
Two black holes at the moment their horizons touch, inside a thin accretion disk,
drawn as a stippled ink image in the style of Luminet's 1979 picture.

Spacetime: two holes on a circular orbit, written as a superposition of two boosted
Kerr-Schild metrics (as in Combi & Ressler 2024, arXiv:2403.13308),

    g = eta + H1 l1 l1 + H2 l2 l2,

where H and l are each hole's Kerr-Schild function and null vector (Kerr, if it
spins), in its own rest frame and boosted to its orbital velocity. By default the
holes are placed at the separation where their coordinate horizons (r' = 2m) touch,
d = 2 m1 + 2 m2 = 2M, and held there: the inspiral during a light ray's transit
is ignored. The orbit is Keplerian, Omega = sqrt(M / d^3), so each equal-mass hole
moves at 0.35c. This is an approximation: superposed metrics are not exact
solutions where the two holes' fields are strong together.

Spins can be tilted from the orbital axis. Then the spins and the orbital plane
precess about the total angular momentum (orbit-averaged 1.5PN spin-orbit equations,
at fixed separation), and the coordinates are turned so that J points along z.

Light: every pixel's ray is traced back from a distant camera through the
time-dependent metric, with the full geodesic (Hamiltonian) equations, so the
holes move while the light crosses them. With non-spinning or aligned spins, the
metric is unchanged by a time shift combined with a rotation (it has a helical
symmetry), so p_t + Omega p_phi is conserved along each ray and p_t is taken from
it. With tilted spins, p_t is integrated, dp_t/dlambda = 1/2 k^a k^b d_t g_ab.

Disk: a thin circumbinary disk in the z = 0 plane (the orbital plane, or with precessing
spins the plane perpendicular to J), from r_in (default 6M, the
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
    python merger.py --set separation=5 --set phase=90 --set r_in=1 --set spin1=0.9 --set tilt1=90
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
    spin1: float = _p(0.0, "spin of the heavier hole, chi = a/m, from -0.99 to 0.99; positive turns with the orbit")
    spin2: float = _p(0.0, "spin of the lighter hole, chi = a/m, from -0.99 to 0.99; positive turns with the orbit")
    tilt1: float = _p(0.0, "angle of the heavier hole's spin from the orbital axis, in degrees (0 = aligned with the "
                           "orbit, 90 = in the orbital plane), as the light passes the holes")
    tilt2: float = _p(0.0, "angle of the lighter hole's spin from the orbital axis, in degrees")
    spin_az1: float = _p(0.0, "direction of the heavier hole's tilted spin in the orbital plane, in degrees, measured "
                              "like phase: 0 = toward screen right, 90 = toward the camera")
    spin_az2: float = _p(0.0, "direction of the lighter hole's tilted spin in the orbital plane, in degrees")
    precess: bool = _p(True, "with tilted spins, let the spins and the orbital plane precess (post-Newtonian "
                             "spin-orbit), with the disk perpendicular to the total angular momentum; False freezes "
                             "the spins and keeps the orbit in the disk plane")
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
            raise ValueError("use spin1 and spin2 for the holes' spins; leave spin at 0")
        if not (abs(self.spin1) <= 0.99 and abs(self.spin2) <= 0.99):
            raise ValueError("spin1 and spin2 must be in [-0.99, 0.99]")
        if not (0 <= self.tilt1 <= 180 and 0 <= self.tilt2 <= 180):
            raise ValueError("tilt1 and tilt2 must be in [0, 180] degrees")
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
def spin_vectors(m, spins, tilts=(0.0, 0.0), azimuths=(0.0, 0.0)):
    """Kerr spin vectors a = chi m of the two holes, in the frame where the orbital
    angular momentum points along +z: tilt is the angle from +z in degrees, azimuth the
    direction in the orbital plane, with the same convention as the phase (0 = screen
    right, 90 = toward the camera)."""
    out = []
    for mi, chi, tilt, az in zip(m, spins, tilts, azimuths):
        th, ph = np.radians(tilt), -np.radians(az)
        out.append(chi * mi * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]))
    return out


class Binary:
    """Two holes on a circular orbit in the z = 0 plane, counterclockwise seen from +z,
    with spins fixed in space. The orbital angle is psi at time t_ref.

    state(i, t) gives hole i's position X, velocity V, acceleration dV, Kerr spin vector
    a and its rate of change da, each (3, N) for the N times t."""
    def __init__(self, d, q, spins=(0.0, 0.0), tilts=(0.0, 0.0), azimuths=(0.0, 0.0), psi=0.0, t_ref=0.0):
        self.m = (1 / (1 + q), q / (1 + q))                 # m1 >= m2, total 1
        self.d = d
        self.omega = np.sqrt(1 / d**3) if d > 0 else 0.0
        self.radius = (d * self.m[1], -d * self.m[0])       # signed distance of each hole from the center of mass
        self.psi0 = psi - self.omega * t_ref                # orbital angle at t = 0
        self.a = spin_vectors(self.m, spins, tilts, azimuths)
        # with both spins along the orbital axis, the spacetime is unchanged by a time shift combined
        # with a rotation about z (a helical symmetry), and p_t + Omega p_phi is conserved along each ray
        self.helical = all(np.hypot(a[0], a[1]) < 1e-12 for a in self.a)

    def state(self, i, t):
        psi = self.psi0 + self.omega * t
        c, s = np.cos(psi), np.sin(psi)
        R, w = self.radius[i], self.omega
        z = np.zeros_like(c)
        X = np.stack([R * c, R * s, z])
        V = np.stack([-R * w * s, R * w * c, z])
        dV = -w * w * X
        a = np.repeat(self.a[i][:, None], c.size, 1)
        return X, V, dV, a, np.zeros_like(a)


class PrecessingBinary:
    """Two holes at a fixed separation whose spins and orbital plane precess.

    The precession follows the orbit-averaged spin-orbit (1.5 post-Newtonian) equations
    (Apostolatos et al. 1994, Kidder 1995), with the separation held fixed:
        dS_i/dt = (2 + 3 m_j / (2 m_i)) / d^3  L x S_i,        dL/dt = -dS_1/dt - dS_2/dt,
    so the total angular momentum J = L + S_1 + S_2 is conserved; L = m1 m2 sqrt(M d) is
    the Newtonian orbital angular momentum. The line between the holes turns at the
    Keplerian rate about L and is carried along as L's direction precesses. Coordinates
    are rotated so that J points along +z; the disk lies in that plane, perpendicular to
    J, while the orbital plane wobbles about it.

    The spins, tilts and azimuths are given at time t_ref, in the frame where L points
    along +z, as in Binary. The track is integrated once on a grid of times and looked
    up with cubic Hermite interpolation."""
    def __init__(self, d, q, spins, tilts, azimuths, psi=0.0, t_ref=0.0, t_range=(-2500.0, 1500.0), dt=0.25):
        self.m = m = (1 / (1 + q), q / (1 + q))
        self.d = d
        self.omega = np.sqrt(1 / d**3)
        self.radius = (d * m[1], -d * m[0])
        self.helical = False
        self.c = (2 + 1.5 * m[1] / m[0], 2 + 1.5 * m[0] / m[1])
        a = spin_vectors(m, spins, tilts, azimuths)
        S = [m[0] * a[0], m[1] * a[1]]
        L = np.array([0.0, 0.0, m[0] * m[1] * np.sqrt(d)])
        n = np.array([np.cos(psi), np.sin(psi), 0.0])
        Rot = _rotation_to_z(L + S[0] + S[1])
        y0 = np.concatenate([Rot @ n, Rot @ L, Rot @ S[0], Rot @ S[1]])
        self.J = Rot @ (L + S[0] + S[1])
        # integrate forward and backward from t_ref
        nf = int(np.ceil((t_range[1] - t_ref) / dt)); nb = int(np.ceil((t_ref - t_range[0]) / dt))
        fwd = self._integrate(y0, dt, nf); bwd = self._integrate(y0, -dt, nb)
        self.y = np.concatenate([bwd[::-1], fwd[1:]])
        self.t0 = t_ref - nb * dt
        self.dt = dt
        der = np.array([self._derivs(y) for y in self.y])   # rows: n, dn, ddn, dS1, dS2
        self.n, self.dn, self.ddn = self.y[:, 0:3], der[:, 0], der[:, 1]
        self.S = (self.y[:, 6:9], self.y[:, 9:12])
        self.dS = (der[:, 2], der[:, 3])
        self.L = self.y[:, 3:6]

    def _rates(self, y):
        n, L, S1, S2 = y[0:3], y[3:6], y[6:9], y[9:12]
        d3 = self.d**3
        dS1 = self.c[0] / d3 * np.cross(L, S1)
        dS2 = self.c[1] / d3 * np.cross(L, S2)
        OmL = (self.c[0] * S1 + self.c[1] * S2) / d3        # dL/dt = OmL x L
        Lh = L / np.linalg.norm(L)
        OmLp = OmL - np.dot(OmL, Lh) * Lh                   # the part that tilts L; the rest would only
        Om = self.omega * Lh + OmLp                         # shift the orbital frequency
        dn = np.cross(Om, n)
        return n, L, Lh, OmL, OmLp, Om, dn, dS1, dS2

    def _f(self, y):
        n, L, Lh, OmL, OmLp, Om, dn, dS1, dS2 = self._rates(y)
        return np.concatenate([dn, np.cross(OmL, L), dS1, dS2])

    def _derivs(self, y):
        n, L, Lh, OmL, OmLp, Om, dn, dS1, dS2 = self._rates(y)
        dLh = np.cross(OmL, Lh)
        dOmL = (self.c[0] * dS1 + self.c[1] * dS2) / self.d**3
        dOmLp = dOmL - (np.dot(dOmL, Lh) + np.dot(OmL, dLh)) * Lh - np.dot(OmL, Lh) * dLh
        dOm = self.omega * dLh + dOmLp
        ddn = np.cross(dOm, n) + np.cross(Om, dn)
        return np.array([dn, ddn, dS1, dS2])

    def _integrate(self, y, h, nsteps):
        out = [y]
        for _ in range(nsteps):
            k1 = self._f(y); k2 = self._f(y + 0.5*h*k1); k3 = self._f(y + 0.5*h*k2); k4 = self._f(y + h*k3)
            y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            out.append(y)
        return np.array(out)

    def _lookup(self, t):
        u = (np.asarray(t, float) - self.t0) / self.dt
        j = np.clip(np.floor(u).astype(int), 0, len(self.y) - 2)
        u = np.clip(u - j, 0.0, 1.0)
        return j, u

    def _hermite(self, f, df, j, u):
        h00 = 2*u**3 - 3*u**2 + 1; h10 = u**3 - 2*u**2 + u; h01 = -2*u**3 + 3*u**2; h11 = u**3 - u**2
        return (h00 * f[j].T + h10 * self.dt * df[j].T + h01 * f[j + 1].T + h11 * self.dt * df[j + 1].T)

    def _linear(self, f, j, u):
        return (1 - u) * f[j].T + u * f[j + 1].T

    def state(self, i, t):
        j, u = self._lookup(t)
        R = self.radius[i]
        n = self._hermite(self.n, self.dn, j, u)
        dn = self._hermite(self.dn, self.ddn, j, u)
        ddn = self._linear(self.ddn, j, u)
        S = self._hermite(self.S[i], self.dS[i], j, u)
        dS = self._linear(self.dS[i], j, u)
        return R * n, R * dn, R * ddn, S / self.m[i], dS / self.m[i]


def _rotation_to_z(v):
    """Rotation matrix taking the direction of v to +z by the smallest turn."""
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    k = np.cross(v, z); s = np.linalg.norm(k); c = np.dot(v, z)
    if s < 1e-14:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = k / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def hole_terms(x, X, V, m, a, derivs=True):
    """Kerr-Schild pieces of one moving hole at points x (3, N): H, l_mu = (l_t, l_s)
    and, if derivs, their spatial derivatives dH (3, N), dlt (3, N) and dls (3, 3, N),
    indexed [j, ...] for d/dx^j.

    In the hole's rest frame (s' = A s, A = 1 + kappa v v^T, kappa = gamma^2/(gamma+1)),
    with Kerr spin vector a (|a| = chi m, pointing along the spin axis; given in the
    rest frame's axes, which a pure boost lines up with the global ones), the Kerr
    metric is eta + H l' l' with r the Kerr radius,
        r^4 - (s'^2 - a^2) r^2 - (a.s')^2 = 0,     H = 2 m r^3 / (r^4 + (a.s')^2),
        l'_s = SIGMA (r^2 s' + r a x s' + (a.s') a) / (r (r^2 + a^2)),   l'_t = 1,
    the outgoing form. For a along +z this is the usual
    l' = (1, SIGMA (r x - a y)/(r^2 + a^2), SIGMA (r y + a x)/(r^2 + a^2), SIGMA z/r).
    Boosting l' back gives l_t = gamma (1 - v.l'_s), l_s = A l'_s - gamma v."""
    s = x - X
    v2 = np.sum(V * V, 0)
    g = 1 / np.sqrt(1 - v2)
    kap = g * g / (g + 1)
    sp = s + kap * np.sum(s * V, 0) * V
    a = np.broadcast_to(a.reshape(3, -1), sp.shape)
    a2 = np.sum(a * a, 0)
    u = np.sum(a * sp, 0)                                    # a . s'
    w = np.sum(sp * sp, 0) - a2
    r2 = 0.5 * (w + np.sqrt(w * w + 4 * u * u))
    r = np.sqrt(r2)
    den = r2 + a2
    Hden = r2 * r2 + u * u
    H = 2 * m * r * r2 / Hden
    axs = np.cross(a, sp, axis=0)
    Nv = r2 * sp + r * axs + u * a
    Q = r * den
    lp = SIGMA * Nv / Q
    lt = g * (1 - np.sum(V * lp, 0))
    ls = lp + kap * np.sum(lp * V, 0) * V - g * V
    out = dict(r=r, H=H, lt=lt, ls=ls)
    if not derivs:
        return out
    # rest-frame gradients d'_k
    D = r * (2 * r2 - w)
    dr = (sp * r2 + u * a) / D
    dHp = 2 * m * (3 * r2 * dr * Hden - r * r2 * (4 * r * r2 * dr + 2 * u * a)) / (Hden * Hden)
    e = np.eye(3)[:, :, None]
    zero = np.zeros_like(r)
    Ax = np.stack([np.stack([zero, a[2], -a[1]]),            # Ax[k, n] = d'_k (a x s')_n = (a x e_k)_n
                   np.stack([-a[2], zero, a[0]]),
                   np.stack([a[1], -a[0], zero])])
    dNv = (2 * r * dr[:, None] * sp[None] + r2 * e + dr[:, None] * axs[None] + r * Ax
           + a[:, None] * a[None])                           # dNv[k, n] = d'_k Nv_n
    dQ = dr * (3 * r2 + a2)
    J = SIGMA * (dNv / Q - Nv[None] * dQ[:, None] / (Q * Q))  # J[k, n] = d'_k l'_n
    # global gradients: d_j = A_jk d'_k (A is symmetric)
    dH = dHp + kap * V * np.sum(V * dHp, 0)
    dlp = J + kap * V[:, None] * np.einsum('kN,knN->nN', V, J)[None]
    vdl = np.einsum('nN,jnN->jN', V, dlp)                   # v . d_j l'_s
    dlt = -g * vdl
    dls = dlp + kap * V[None] * vdl[:, None]
    out.update(dH=dH, dlt=dlt, dls=dls)
    return out


def time_derivs(x, X, V, dV, m, a, da, h, eps=0.1):
    """d/dt of H, l_t and l_s at fixed x for a hole moving at V, accelerating at dV, whose
    spin vector changes at da; h holds the hole's terms with spatial derivatives. The
    hole's motion gives -V.grad; the turning of its velocity and spin is taken by a
    central difference in those directions (at fixed position), which is accurate to
    eps^2 times their (small) curvature."""
    dtH = -np.sum(V * h['dH'], 0)
    dtlt = -np.sum(V * h['dlt'], 0)
    dtls = -np.einsum('jN,jnN->nN', V, h['dls'])
    if np.any(dV) or np.any(da):
        hp = hole_terms(x, X, V + eps * dV, m, a + eps * da, derivs=False)
        hm = hole_terms(x, X, V - eps * dV, m, a - eps * da, derivs=False)
        dtH = dtH + (hp['H'] - hm['H']) / (2 * eps)
        dtlt = dtlt + (hp['lt'] - hm['lt']) / (2 * eps)
        dtls = dtls + (hp['ls'] - hm['ls']) / (2 * eps)
    return dtH, dtlt, dtls


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
        kl = k_t * h['lt'] + np.sum(k_s * h['ls'], 0)
        kdl = k_t * h['dlt'] + np.einsum('mN,jmN->jN', k_s, h['dls'])
        F = F + 0.5 * h['dH'] * kl * kl + h['H'] * kl * kdl
    return F


def metric_terms(binary, t, x, derivs=True, dt=False):
    """Both holes' Kerr-Schild terms at times t and points x; with dt, also their time
    derivatives at fixed x (dtH, dtlt, dtls)."""
    T = []
    for i in (0, 1):
        X, V, dV, a, da = binary.state(i, t)
        h = hole_terms(x, X, V, binary.m[i], a, derivs)
        if dt:
            h['dtH'], h['dtlt'], h['dtls'] = time_derivs(x, X, V, dV, binary.m[i], a, da, h)
        T.append(h)
    return T


def force_t(k_t, k_s, T):
    """dp_t/dlambda = 1/2 k^a k^b d_t g_ab, as in force."""
    F = 0.0
    for h in T:
        kl = k_t * h['lt'] + np.sum(k_s * h['ls'], 0)
        kdl = k_t * h['dtlt'] + np.sum(k_s * h['dtls'], 0)
        F = F + 0.5 * h['dtH'] * kl * kl + h['H'] * kl * kdl
    return F


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


def trace_rays(p, binary, X, Y, constraint=False):
    """Trace a batch of rays backward from the camera. Returns, per ray, the disk
    radius where it first meets the disk (NaN if never), how many times it had
    crossed the disk plane before that, the redshift 1+z of the light, and which
    hole swallowed it (0 = heavier, 1 = lighter, 2 = still orbiting at max_steps, -1 = none).
    With constraint=True it also returns the largest |g^ab p_a p_b| met along each
    ray, relative to p_t^2 + |p_s|^2; it stays ~0 if the equations of motion are integrated consistently.

    When the binary has a helical symmetry, p_t comes from the conserved p_t + Omega p_phi;
    otherwise p_t is integrated, dp_t/dlambda = 1/2 k^a k^b d_t g_ab."""
    helical = binary.helical
    N = X.size
    x, t, p_s, p_t, E_cam = initial_rays(p, binary, X, Y)
    w = binary.omega if helical else 0.0
    Q = p_t + w * (x[0] * p_s[1] - x[1] * p_s[0])            # helical: conserved; otherwise p_t itself
    rho_hit = np.full(N, np.nan); n_cross = np.zeros(N, int); opz = np.full(N, np.nan)
    captor = np.full(N, -1, np.int8)                        # which hole swallowed the ray (-1: none)
    cons = np.zeros(N)
    idx = np.arange(N)
    crossings = np.zeros(N, int)
    m = binary.m
    D = p.camera_distance

    def pt_of(x, ps, Q):
        return Q - w * (x[0] * ps[1] - x[1] * ps[0])

    def rhs(t, x, ps, Q):
        pt = pt_of(x, ps, Q)
        T = metric_terms(binary, t, x, dt=not helical)
        kt, ks = raise_index(pt, ps, T)
        dQ = 0.0 if helical else force_t(kt, ks, T)
        return kt, ks, force(kt, ks, T), dQ, T

    for step in range(p.max_steps):
        kt1, ks1, F1, dQ1, T = rhs(t, x, p_s, Q)
        r_near = np.minimum(T[0]['r'] / (2 * m[0]), T[1]['r'] / (2 * m[1]))
        r_hole = np.minimum(T[0]['r'], T[1]['r'])
        # fallen in: inside r = m, between a Kerr hole's two horizons (r = m +- sqrt(m^2 - a^2)), or where
        # the summed metric breaks down
        done = (r_near < 0.5) | (lorentzian_margin(T) < MIN_MARGIN) | ~np.isfinite(kt1)
        out = (np.sum(x * x, 0) > (1.05 * D)**2)            # escaped past the camera's distance
        keep = ~(done | out)
        if not keep.all():
            captor[idx[done]] = np.where(T[0]['r'][done] / m[0] <= T[1]['r'][done] / m[1], 0, 1)
            idx, x, t, p_s, Q, crossings = idx[keep], x[:, keep], t[keep], p_s[:, keep], Q[keep], crossings[keep]
            kt1, ks1, F1 = kt1[keep], ks1[:, keep], F1[:, keep]
            if not helical:
                dQ1 = dQ1[keep]
            r_hole = r_hole[keep]
            if len(idx) == 0:
                break
        if constraint:                                      # checked on rays still in flight
            pt1 = pt_of(x, p_s, Q)
            c = np.abs(kt1 * pt1 + np.sum(ks1 * p_s, 0)) / (pt1 * pt1 + np.sum(p_s * p_s, 0))
            cons[idx] = np.maximum(cons[idx], c)
        # step size ~ distance to the nearer hole, coarser far away; the light's coordinate speed
        # grows near the holes, so scale by it to keep each step a small fraction of the distance
        frac = p.step * (1 + np.clip((r_hole - 30) / 30, 0, 4))
        speed = np.maximum(np.sqrt(np.sum(ks1 * ks1, 0)), np.abs(kt1))
        h = -frac * r_hole / speed                          # negative: backward along the ray
        kt2, ks2, F2, dQ2, _ = rhs(t + 0.5*h*kt1, x + 0.5*h*ks1, p_s + 0.5*h*F1, Q + 0.5*h*dQ1)
        kt3, ks3, F3, dQ3, _ = rhs(t + 0.5*h*kt2, x + 0.5*h*ks2, p_s + 0.5*h*F2, Q + 0.5*h*dQ2)
        kt4, ks4, F4, dQ4, _ = rhs(t + h*kt3, x + h*ks3, p_s + h*F3, Q + h*dQ3)
        t_new = t + h/6 * (kt1 + 2*kt2 + 2*kt3 + kt4)
        x_new = x + h/6 * (ks1 + 2*ks2 + 2*ks3 + ks4)
        p_new = p_s + h/6 * (F1 + 2*F2 + 2*F3 + F4)
        Q_new = Q + h/6 * (dQ1 + 2*dQ2 + 2*dQ3 + dQ4)
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
                Qc = Q[j] + f[hit] * (Q_new[j] - Q[j])
                xc = xc[:, hit]
                rho_hit[idx[j]] = rho[hit]
                n_cross[idx[j]] = crossings[j]
                opz[idx[j]] = emitted_energy(binary, tc, xc, pc, pt_of(xc, pc, Qc)) / E_cam[idx[j]]
            crossings[i] += 1
            # rays that met the disk stop there (it is opaque)
            if hit.any():
                keep = np.ones(len(idx), bool); keep[i[hit]] = False
                idx, x_new, t_new, p_new, Q_new, crossings = (idx[keep], x_new[:, keep], t_new[keep],
                                                              p_new[:, keep], Q_new[keep], crossings[keep])
        x, t, p_s, Q = x_new, t_new, p_new, Q_new
        if len(idx) == 0:
            break
    captor[idx] = 2                                         # still orbiting at the step limit
    if constraint:
        return rho_hit, n_cross, opz, captor, cons
    return rho_hit, n_cross, opz, captor


def emitted_energy(binary, t, x, p_s, p_t):
    """The light's energy -p.u measured by the disk gas at x; divided by the energy
    the camera measures, it gives 1+z. The gas orbits about the z axis at the
    Keplerian rate of the total mass, Omega_d = rho^-3/2, which is exact for a
    single Schwarzschild hole in Kerr-Schild coordinates."""
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
def light_arrival_time(p, d, q):
    """Time at which light from the camera, passing near the center, crosses the plane
    through the center facing the camera (traced on a non-spinning binary: about the
    camera distance plus the Shapiro delay)."""
    binary = Binary(d, q)
    n, _, _ = camera_frame(p.inclination)
    x, t, p_s, p_t, _ = initial_rays(p, binary, np.array([3.0 * max(d, 2)]), np.array([0.0]))
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
            return float(t_new[0])
        x, t = x_new, t_new
    raise RuntimeError("reference ray never reached the center")


def make_binary(p):
    """The binary for parameters p, with the holes at orbital phase p.phase (in the
    orbital plane) as the light passes them. psi is the heavier hole's angle from +x
    (screen right); the camera is toward -y, so phase 0 (psi = 0) is side by side and
    phase 90 (psi = -90 deg) puts it in front."""
    d, q = p.separation, p.mass_ratio
    t_ref = light_arrival_time(p, d, q) if d > 0 else 0.0
    spin = dict(spins=(p.spin1, p.spin2), tilts=(p.tilt1, p.tilt2), azimuths=(p.spin_az1, p.spin_az2),
                psi=np.radians(-p.phase), t_ref=t_ref)
    binary = Binary(d, q, **spin)
    if not binary.helical and p.precess and d > 0:
        D = p.camera_distance
        binary = PrecessingBinary(d, q, **spin, t_range=(-2.5 * D, 1.5 * D))
    return binary


def binary_image(p, verbose=True, workers=None):
    """Trace every image cell and return the dict blackhole.stipple expects."""
    VIEW, VIEW_H = p.view, p.view_h
    W = p.grid_w; H = int(round(W / p.aspect))
    xs = np.linspace(-VIEW, VIEW, W); ys = np.linspace(-VIEW_H, VIEW_H, H)
    X, Y = np.meshgrid(xs, ys)
    binary = make_binary(p)
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
        parts = [trace_rays(p, binary, Xf[c], Yf[c]) for c in chunks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(trace_rays, [p] * len(chunks), [binary] * len(chunks),
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
        if k in ('separation', 'mass_ratio', 'spin1', 'spin2', 'tilt1', 'tilt2', 'spin_az1', 'spin_az2', 'phase', 'camera_distance', 'step', 'center_x', 'center_y'):
            return float(v)
        if k == 'precess':
            return v if isinstance(v, bool) else str(v).lower() in ('1', 'true', 'yes', 'on')
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
