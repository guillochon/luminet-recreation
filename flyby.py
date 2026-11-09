"""
Fly past a black hole on a parabolic orbit, filmed as a stippled ink drawing.

The camera falls in from far away on a parabolic geodesic (energy exactly 1)
whose plane is tilted to the disk, swings past the hole and leaves again,
always aimed at the hole. Every frame is ray traced from the camera's actual
position and velocity, so the view includes aberration and the Doppler shift
of the camera's own motion as well as the hole's light bending.

The ink dots are attached to the disk: each is a point on the disk that orbits
with the gas, so the dots move smoothly as the disk turns and the view changes.
Each frame, dots that end up too crowded fade out and gaps are filled with new
dots that fade in, so the dot spacing still follows the brightness, as in the
stills.

    python flyby.py --set spin=0.998                 # full fly-by video
    python flyby.py --set spin=0.998 --frames 0.5    # one frame at periapsis, as PNG
    python flyby.py --set spin=0.998 --seconds 3     # a short clip from the start
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import blackhole as bh

VIEW = 36.0               # screen half-width, in the same units as the stills, so dot spacing matches them
DPHI = 0.01               # ray integration step, in radians of orbital angle (RK4)
MAX_STEPS = 1400


# --- Kerr geometry ---------------------------------------------------------------
def metric_parts(a, r, th):
    """Sigma, Delta and the ZAMO (zero-angular-momentum observer) lapse alpha,
    frame-dragging rate omega and cylindrical radius varpi."""
    s, c = np.sin(th), np.cos(th)
    Sig = r*r + a*a*c*c
    Del = r*r - 2*r + a*a
    A = (r*r + a*a)**2 - a*a*Del*s*s
    return Sig, Del, np.sqrt(Sig*Del/A), 2*a*r/A, np.sqrt(A/Sig)*s


# --- Camera orbit ----------------------------------------------------------------
def orbit_constants(a, r_p, tilt_deg, prograde=True):
    """Angular momentum L_z and Carter constant Q of the parabolic (E = 1) orbit with
    periapsis r_p whose plane is tilted tilt_deg from the disk: cos(tilt) = L_z / sqrt(L_z^2 + Q)."""
    ci = np.cos(np.radians(tilt_deg)) * (1 if prograde else -1)
    si = np.sin(np.radians(tilt_deg))
    def R(r, L):
        Lz, Q = L*ci, (L*si)**2
        return ((r*r + a*a) - a*Lz)**2 - (r*r - 2*r + a*a)*(r*r + (Lz - a)**2 + Q)
    lo, hi = 0.0, 50.0                                  # R(r_p) > 0 at L = 0; find where it reaches 0
    for _ in range(200):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if R(r_p, mid) > 0 else (lo, mid)
    L = 0.5*(lo + hi)
    rr = np.linspace(r_p*1.0001, 2000, 20000)
    if not (R(rr, L) > 0).all():
        raise ValueError(f"no parabolic orbit with periapsis {r_p}M: it would be captured")
    return L*ci, (L*si)**2


def orbit_equations(a, Lz, Q):
    """Mino-time equations of motion of the camera's (E = 1) orbit, for the state
    (r, r', u = cos(theta), u', phi, t, tau)."""
    def deriv(s):
        r, vr, u, vu = s[0], s[1], s[2], s[3]
        P = r*r + a*a - a*Lz
        K = r*r + (Lz - a)**2 + Q
        Del = r*r - 2*r + a*a
        s2 = max(1 - u*u, 1e-12)
        return np.array([vr, 0.5*(4*r*P - (2*r - 2)*K - 2*r*Del),        # r'' = R'(r)/2
                         vu, -(Q + Lz*Lz)*u,                              # u'' = U'(u)/2
                         -(a - Lz/s2) + a*P/Del,                          # phi'
                         -a*(a*s2 - Lz) + (r*r + a*a)*P/Del,              # t'
                         r*r + a*a*u*u])                                  # tau' = Sigma
    return deriv


def camera_path(a, r_p, tilt, prograde, r_far):
    """The whole flight: from r_far in to periapsis (where the camera is also at its
    highest point above the disk) and back out to r_far."""
    Lz, Q = orbit_constants(a, r_p, tilt, prograde)
    deriv = orbit_equations(a, Lz, Q)
    u_max = np.sqrt(Q / (Q + Lz*Lz))                    # = sin(tilt): highest point above the disk
    s = np.array([r_p, 0.0, u_max, 0.0, 0.0, 0.0, 0.0])
    rows = [s.copy()]
    while s[0] < r_far:
        h = 0.02 / (s[0]**2 + a*a*s[2]**2)              # ~0.02M of proper time per step
        k1 = deriv(s); k2 = deriv(s + 0.5*h*k1); k3 = deriv(s + 0.5*h*k2); k4 = deriv(s + h*k3)
        s = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
        rows.append(s.copy())
    out = np.array(rows)                                 # columns: r, r', u, u', phi, t, tau
    # The orbit is time-symmetric about periapsis: the inbound half mirrors the outbound one.
    inbound = out[:0:-1] * [1, -1, 1, -1, -1, -1, -1]
    full = np.vstack([inbound, out])
    return dict(Lz=Lz, Q=Q, r=full[:, 0], vr=full[:, 1], u=full[:, 2], vu=full[:, 3],
                phi=full[:, 4], t=full[:, 5], tau=full[:, 6])


def camera_at(path, tau):
    return {k: float(np.interp(tau, path['tau'], path[k])) for k in ('r', 'vr', 'u', 'vu', 'phi', 't')}


def camera_velocity(a, cam, Lz):
    """The camera's velocity in the local ZAMO frame, as (v_r, v_theta, v_phi)."""
    r, u = cam['r'], cam['u']
    th = np.arccos(u)
    Sig, Del, alpha, omega, varpi = metric_parts(a, r, th)
    dr = cam['vr'] / Sig
    dth = -cam['vu'] / (np.sin(th) * Sig)
    ur, uth, uph = np.sqrt(Sig/Del)*dr, np.sqrt(Sig)*dth, Lz/varpi
    gam = (1 - omega*Lz) / alpha
    return np.array([ur, uth, uph]) / gam, gam, (gam*gam - 1 - ur*ur - uth*uth - uph*uph)


def aberrate(k, v):
    """Direction, in a frame moving at v, of a photon moving along unit vector k."""
    vv = np.dot(v, v)
    if vv == 0:
        return k
    g = 1 / np.sqrt(1 - vv)
    kv = k @ v
    return (k + ((g - 1)*kv/vv - g)*v) / (g*(1 - kv))


def camera_rays(a, cam, v, X, Y, fov_deg):
    """For screen points (X, Y) in [-VIEW, VIEW] x [-VIEW/aspect, ...], the photon
    constants and the backward-ray starting state at the camera."""
    r, u = cam['r'], cam['u']
    th = np.arccos(u)
    Sig, Del, alpha, omega, varpi = metric_parts(a, r, th)
    # camera axes, in the camera's rest frame (components along the ZAMO r, theta, phi axes):
    # aimed at the hole, with "up" toward the hole's north pole
    fwd = -aberrate(np.array([1.0, 0, 0]), v)          # light from the hole's direction moves along +e_r
    up = -aberrate(np.array([0, 1.0, 0]), v)           # light from the north moves along +e_theta
    fwd /= np.linalg.norm(fwd)
    up = up - (up @ fwd) * fwd; up /= np.linalg.norm(up)
    right = np.cross(fwd, up)
    T = np.tan(np.radians(fov_deg) / 2) / VIEW
    D = fwd + (X.ravel()*T)[:, None]*right + (Y.ravel()*T)[:, None]*up
    k = -D / np.linalg.norm(D, axis=1, keepdims=True)  # photon direction in the camera frame (energy 1)
    # boost to the ZAMO frame
    vv = v @ v
    g = 1 / np.sqrt(1 - vv)
    kv = k @ v
    pt = g*(1 + kv)
    p = k + np.outer(((g - 1)*kv/vv + g) if vv > 0 else 0*kv, v)
    pr, pth, pph = p[:, 0], p[:, 1], p[:, 2]
    E = alpha*pt + omega*varpi*pph                      # -p_t
    L = varpi*pph                                       # p_phi
    Qc = (np.sqrt(Sig)*pth)**2 + u*u*(L*L/(1 - u*u) - a*a*E*E)
    ell, eta = L/E, Qc/(E*E)
    x0 = 1/r
    drdl = np.sqrt(Sig*Del)*pr/E                        # photon's dr/dlambda (Mino time, per unit energy)
    dthdl = np.sqrt(Sig)*pth/E
    return dict(ell=ell, eta=eta, E=E, x0=x0, u0=u, vx=x0*x0*drdl, vu=np.sin(th)*dthdl)   # backward ray


# --- Ray tracing (runs in worker processes) ----------------------------------------
def trace_rays(a, x0, u0, vx, vu, ell, eta, crossings=2):
    """Trace backward rays from the camera; for each of the first `crossings` passes
    through the disk plane return r there and how much the photon's phi (modulo 2 pi) and t
    advanced between there and the camera. Same Mino-time scheme as
    blackhole._trace_kerr_rays, which starts from a distant observer instead."""
    n = len(ell)
    A = a*a - a*ell
    Q = eta + (ell - a)**2
    out = np.full((3, crossings, n), np.nan)            # r, delta phi, delta t
    idx = np.arange(n)
    x = np.full(n, x0); vx = vx.copy()
    # The angular motion (theta, and the l / sin^2 theta part of phi) is tracked as a
    # point N = (sin th cos ph, sin th sin ph, cos th) moving on the unit sphere:
    # in Mino time it obeys N'' = a^2 u z - (a^2 u^2 + |N'|^2) N with u = N_z, which,
    # unlike dphi/dlambda = l / (1 - u^2), has no singularity at the poles. (Rays that
    # pass close to a pole otherwise get a wrong phi, in a thin line on screen.)
    s0 = np.sqrt(1 - u0*u0)
    N = np.zeros((n, 3)); N[:, 0] = s0; N[:, 2] = u0
    V = np.zeros((n, 3)); V[:, 0] = -u0*vu/s0; V[:, 1] = -ell/s0; V[:, 2] = vu   # backward ray: phi runs backward
    Ip = np.zeros(n); It = np.zeros(n)
    h = DPHI / np.maximum(np.sqrt(np.maximum(Q, 0)), 1.0)
    hits = np.zeros(n, int)
    x_h = 1 / (1 + np.sqrt(1 - a*a))
    def acc(x, N, V, A, Q):
        u = N[:, 2]
        aN = -(a*a*u*u + (V*V).sum(1))[:, None] * N
        aN[:, 2] += a*a*u
        return 2*A*x*(1 + A*x*x) - Q*x*(1 - 3*x + 2*a*a*x*x), aN
    def rates(x, u, A, ell):                            # the regular parts of dphi/dlambda and dt/dlambda
        s2 = 1 - u*u
        D = 1 - 2*x + a*a*x*x
        B = 1 + A*x*x
        return -a + a*B/D, -a*(a*s2 - ell) + (1 + a*a*x*x)*B/(x*x*D)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for step in range(MAX_STEPS):
            if step % 20 == 0:
                live = (hits < crossings) & (x < x_h) & ~((x < 0) & (vx < 0))
                if not live.all():
                    idx, x, vx, N, V, h, A, Q, hits, Ip, It, ell = (
                        w[live] for w in (idx, x, vx, N, V, h, A, Q, hits, Ip, It, ell))
                if len(idx) == 0:
                    break
            hc = h[:, None]
            k1x, k1N = acc(x, N, V, A, Q)
            x2, N2 = x + 0.5*h*vx, N + 0.5*hc*V
            x2v, N2v = vx + 0.5*h*k1x, V + 0.5*hc*k1N
            k2x, k2N = acc(x2, N2, N2v, A, Q)
            x3, N3 = x + 0.5*h*x2v, N + 0.5*hc*N2v
            x3v, N3v = vx + 0.5*h*k2x, V + 0.5*hc*k2N
            k3x, k3N = acc(x3, N3, N3v, A, Q)
            x4, N4 = x + h*x3v, N + hc*N3v
            x4v, N4v = vx + h*k3x, V + hc*k3N
            k4x, k4N = acc(x4, N4, N4v, A, Q)
            p1, t1 = rates(x, N[:, 2], A, ell); p2, t2 = rates(x2, N2[:, 2], A, ell)
            p3, t3 = rates(x3, N3[:, 2], A, ell); p4, t4 = rates(x4, N4[:, 2], A, ell)
            Ip_new = Ip + h/6*(p1 + 2*p2 + 2*p3 + p4)
            It_new = It + h/6*(t1 + 2*t2 + 2*t3 + t4)
            x_new = x + h/6*(vx + 2*x2v + 2*x3v + x4v)
            N_new = N + hc/6*(V + 2*N2v + 2*N3v + N4v)
            vx = vx + h/6*(k1x + 2*k2x + 2*k3x + k4x)
            V = V + hc/6*(k1N + 2*k2N + 2*k3N + k4N)
            u, u_new = N[:, 2], N_new[:, 2]
            crossed = ((u > 0) != (u_new > 0)) & (x > 0) & (x < x_h) & (hits < crossings)
            if crossed.any():
                i = np.nonzero(crossed)[0]
                f = u[i] / (u[i] - u_new[i])
                Nc = N[i] + f[:, None]*(N_new[i] - N[i])
                out[0, hits[i], idx[i]] = 1 / (x[i] + f*(x_new[i] - x[i]))
                # forward phi change: the regular part, plus the sphere point's azimuth (the
                # backward ray started at azimuth 0); only phi modulo 2 pi is used
                out[1, hits[i], idx[i]] = Ip[i] + f*(Ip_new[i] - Ip[i]) - np.arctan2(Nc[:, 1], Nc[:, 0])
                out[2, hits[i], idx[i]] = It[i] + f*(It_new[i] - It[i])
                hits[i] += 1
            x, N, Ip, It = x_new, N_new, Ip_new, It_new
    return out


def trace_frame(pool, a, rays, chunks=32):
    parts = np.array_split(np.arange(len(rays['ell'])), chunks)
    futs = [pool.submit(trace_rays, a, rays['x0'], rays['u0'], rays['vx'][c], rays['vu'][c],
                        rays['ell'][c], rays['eta'][c]) for c in parts]
    return parts, futs


def collect(parts, futs, n):
    out = np.full((3, 2, n), np.nan)
    for c, f in zip(parts, futs):
        out[:, :, c] = f.result()
    return out


# --- What the camera sees -------------------------------------------------------------
def emitted_flux(p, r):
    F = bh.page_thorne if p.spin == 0 else (lambda rr: bh.novikov_thorne(rr, p.spin))
    if p.sharp_inner_edge:
        return (1 - np.exp(-np.maximum(r - p.r_in, 0) / p.edge_softness)) * F(np.maximum(r, p.r_peak))
    return F(r)


def frame_fields(p, cam, rays, traced, shape):
    """Per screen point: which image of the disk it shows (0 = direct, 1 = secondary,
    -1 = none), the disk radius and azimuth there, the time the light left the
    disk, and the observed brightness."""
    a = p.spin
    layer = np.full(shape[0]*shape[1], -1)
    R = np.full(layer.shape, np.nan); PHI = R.copy(); TEM = R.copy(); F = np.zeros(layer.shape)
    for n in (1, 0):                                    # direct image overwrites the secondary (opaque disk)
        r, dphi, dt = traced[0, n], traced[1, n], traced[2, n]
        hit = (r >= p.r_in) & (r <= p.r_out)
        rr = np.where(hit, r, 10.0)
        om = 1 / (rr**1.5 + a)
        ut = 1 / np.sqrt(1 - 2/rr + 4*a*om/rr - (rr*rr + a*a + 2*a*a/rr)*om*om)
        one_plus_z = rays['E'] * ut * (1 - om*rays['ell'])
        flux = emitted_flux(p, rr) / one_plus_z**4
        phi_em = cam['phi'] - dphi                       # where on the disk, and when, the light left
        t_em = cam['t'] - dt
        layer = np.where(hit, n, layer); R = np.where(hit, r, R)
        PHI = np.where(hit, phi_em, PHI); TEM = np.where(hit, t_em, TEM); F = np.where(hit, flux, F)
    return tuple(v.reshape(shape) for v in (layer, R, PHI, TEM, F))


def exposure_reference(p, F, layer):
    """The auto-exposure reference brightness (see blackhole.stipple), for one frame."""
    f = F[layer >= 0]
    if f.size == 0:
        return np.nan
    full = (p.dot_cap / p.dot_gain) ** (1 / p.dot_gamma)
    return min(f.max(), np.quantile(f, 1 - bh.LUMINET_SATURATED_FRACTION) / full)


def density_field(p, R, F, layer, ref):
    weight = np.ones_like(R)
    if p.banded:
        k = np.log(R / p.r_in) / np.log(p.ring_step)
        weight = np.where(np.isnan(R), 1.0, 1 + p.band_depth * np.cos(2*np.pi*np.nan_to_num(k)))
    d = np.minimum(p.dot_gain * np.nan_to_num((F / ref)**p.dot_gamma), p.dot_cap) * weight
    d = d + np.where(layer >= 0, p.background_density, 0.0)
    return np.where(layer >= 0, np.minimum(d, 1.0), 0.0)


# --- Dots attached to the disk ---------------------------------------------------------
FADE = 8                  # frames for a dot to fade in or out
KEEP = 0.8                # an existing dot fades out only if an older one is closer than KEEP x their spacing
FILL_OVERSAMPLE = 3       # candidate points per expected dot when filling gaps


class Dots:
    """Ink dots attached to the disk. Each dot is labeled by its radius and by the
    azimuth its piece of gas had at t = 0; gas on circular orbits keeps both, so
    the dot's current azimuth is that label + Omega(r) t."""
    FIELDS = ('lay', 'r', 'phi0', 'born', 'rank', 'size', 'ang', 'off', 'dying')

    def __init__(self, p, seed, rotation=1.0, layers=(0, 1)):
        self.p = p
        self.rotation = rotation                         # 1 = the physical rotation speed
        self.layers = layers                             # image layers (0 = direct, 1 = secondary, ...)
        self.rng = np.random.default_rng(seed)
        self.spacing = p.dot_spacing * VIEW / 30
        for k in self.FIELDS:
            setattr(self, k, np.zeros(0, int if k in ('lay', 'born', 'dying') else float))

    def omega(self, r):
        return self.rotation / (r**1.5 + self.p.spin)    # Keplerian angular velocity

    def _take(self, keep):
        for k in self.FIELDS:
            setattr(self, k, getattr(self, k)[keep])

    def jacobians(self, layer, Lx, Ly):
        """Per image layer: how the disk-plane position changes per screen pixel
        (d/dj, d/di of Lx and Ly), using only pixels of that layer."""
        J = {}
        for n in self.layers:
            ok = layer == n
            Lxn, Lyn = np.where(ok, Lx, np.nan), np.where(ok, Ly, np.nan)
            J[n] = (np.gradient(Lxn, axis=1), np.gradient(Lxn, axis=0),
                    np.gradient(Lyn, axis=1), np.gradient(Lyn, axis=0))
        return J

    def project(self, grid, layer, Lx, Ly, t_ref, J):
        """Screen position of every dot (NaN where its piece of the disk is hidden or
        off screen). Lx, Ly give, for each pixel, where its piece of gas is at time
        t_ref; find the pixel showing the gas the dot sits on."""
        xs, ys = grid
        H, W = layer.shape
        dx, dy = xs[1] - xs[0], ys[1] - ys[0]
        pos = np.full((len(self.r), 2), np.nan)
        for n in self.layers:
            ok = layer == n
            mine = np.nonzero(self.lay == n)[0]
            if not ok.any() or len(mine) == 0:
                continue
            Jxj, Jxi, Jyj, Jyi = J[n]
            ii, jj = np.nonzero(ok)
            tree = cKDTree(np.column_stack([Lx[ok], Ly[ok]]))
            r = self.r[mine]
            phi = self.phi0[mine] + self.omega(r) * t_ref
            q = np.column_stack([r*np.cos(phi), r*np.sin(phi)])
            _, k = tree.query(q)
            pi, pj = ii[k], jj[k]
            # solve with the nearest pixel's local linear map, and with its 8 neighbors';
            # keep whichever puts the point inside that pixel's own cell
            best = np.full(len(mine), np.inf); bj = np.zeros(len(mine)); bi = np.zeros(len(mine))
            ci = pi.copy(); cj = pj.copy()
            for oi in (0, -1, 1):
                for oj in (0, -1, 1):
                    ti, tj = np.clip(pi + oi, 0, H - 1), np.clip(pj + oj, 0, W - 1)
                    a11, a12, a21, a22 = Jxj[ti, tj], Jxi[ti, tj], Jyj[ti, tj], Jyi[ti, tj]
                    ex, ey = q[:, 0] - Lx[ti, tj], q[:, 1] - Ly[ti, tj]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        det = a11*a22 - a12*a21
                        dj = (a22*ex - a12*ey) / det
                        di = (-a21*ex + a11*ey) / det
                    off = np.maximum(np.abs(dj), np.abs(di))
                    off = np.where(np.isfinite(off) & (layer[ti, tj] == n), off, np.inf)
                    better = off < best
                    best = np.where(better, off, best); bj = np.where(better, dj, bj); bi = np.where(better, di, bi)
                    ci = np.where(better, ti, ci); cj = np.where(better, tj, cj)
            good = best <= 0.75
            pos[mine[good], 0] = xs[cj[good]] + bj[good]*dx
            pos[mine[good], 1] = ys[ci[good]] + bi[good]*dy
        return pos

    def label(self, grid, layer, Lx, Ly, t_ref, J, pts):
        """Layer, radius and t = 0 azimuth of the disk under screen points pts:
        the exact inverse of project()."""
        xs, ys = grid
        H, W = layer.shape
        j = np.clip(np.round((pts[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int), 0, W - 1)
        i = np.clip(np.round((pts[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int), 0, H - 1)
        fj = (pts[:, 0] - xs[j]) / (xs[1] - xs[0]); fi = (pts[:, 1] - ys[i]) / (ys[1] - ys[0])
        lay = layer[i, j]
        lx = np.full(len(pts), np.nan); ly = lx.copy()
        for n in self.layers:
            m = lay == n
            Jxj, Jxi, Jyj, Jyi = J[n]
            lx[m] = Lx[i[m], j[m]] + Jxj[i[m], j[m]]*fj[m] + Jxi[i[m], j[m]]*fi[m]
            ly[m] = Ly[i[m], j[m]] + Jyj[i[m], j[m]]*fj[m] + Jyi[i[m], j[m]]*fi[m]
        r = np.hypot(lx, ly)
        good = (lay >= 0) & np.isfinite(r)
        phi0 = np.arctan2(ly, lx) - self.omega(np.where(good, r, 10.0)) * t_ref   # wind back from t_ref to t = 0
        return good, lay, r, phi0

    def step(self, f, grid, layer, R, PHI, TEM, dens, t_ref):
        """Advance to frame f. Returns screen positions, sizes, lobe angles and offsets of the dots to draw.

        Dot spacing works like the stills, with a pair of dots counted as too close
        when nearer than the average of their two spacing radii. An existing dot
        fades out only when an older one is closer than KEEP x that, while a new dot
        needs a full spacing radius: the gap keeps dots from flickering in and out."""
        xs, ys = grid
        # Dots ride with the gas as it is at the camera's time t_ref. The light from
        # different pixels left the disk at different times (TEM), and near the hole
        # the gas moves so fast that, allowing for that, the same parcel can appear
        # twice; the brightness keeps all of that physics, but the dot pattern
        # ignores the delays, so each parcel has one place on screen.
        Lx, Ly = R*np.cos(PHI), R*np.sin(PHI)
        J = self.jacobians(layer, Lx, Ly)
        # 1) project every dot; drop those now hidden or off screen
        pos = self.project(grid, layer, Lx, Ly, t_ref, J)
        visible = ~np.isnan(pos[:, 0])
        self.stats = dict(lost=int((~visible & (self.dying == 0)).sum()), faded=0)
        self._take(visible); pos = pos[visible]
        radius = self.spacing / np.sqrt(np.maximum(sample(dens, xs, ys, pos), 1e-4))
        self.dying[(radius > self.spacing / np.sqrt(1e-3)) & (self.dying == 0)] = FADE   # no ink wanted here any more
        # 2) crowded pairs: the younger dot fades out
        alive = np.nonzero(self.dying == 0)[0]
        if len(alive) > 1:
            pa, ra = pos[alive], radius[alive]
            nb = cKDTree(pa).query_ball_point(pa, r=KEEP * ra * 1.5)
            ii = np.repeat(np.arange(len(pa)), [len(x) for x in nb]); jj = np.concatenate(nb).astype(int)
            close = (ii < jj) & (np.hypot(*(pa[ii] - pa[jj]).T) < KEEP * 0.5 * (ra[ii] + ra[jj]))
            ii, jj = ii[close], jj[close]
            if len(ii):
                prio = np.lexsort((self.rank[alive], self.born[alive]))        # older first
                rank_of = np.empty(len(pa), int); rank_of[prio] = np.arange(len(pa))
                old, young = np.where(rank_of[ii] < rank_of[jj], ii, jj), np.where(rank_of[ii] < rank_of[jj], jj, ii)
                order = np.argsort(rank_of[old], kind='stable')
                killed = np.zeros(len(pa), bool)
                for o, y in zip(old[order], young[order]):
                    if not killed[o]:
                        killed[y] = True
                self.dying[alive[killed]] = FADE
                self.stats['faded'] = int(killed.sum())
        # 3) fill gaps with new dots, placed like the stills: Poisson-disk sampling
        alive = self.dying == 0
        oversample = FILL_OVERSAMPLE if alive.any() else self.p.oversample     # the first frame starts from scratch, like a still
        n_cand = int(oversample * dens.mean() * (2*VIEW) * (2*ys[-1]) / self.spacing**2)
        cx = self.rng.uniform(xs[0], xs[-1], n_cand); cy = self.rng.uniform(ys[0], ys[-1], n_cand)
        cd = sample(dens, xs, ys, np.column_stack([cx, cy]))
        keep = self.rng.random(n_cand) < cd
        cand = np.column_stack([cx[keep], cy[keep]]); cr = self.spacing / np.sqrt(np.maximum(cd[keep], 1e-4))
        if alive.sum() and len(cand):
            pa, ra = pos[alive], radius[alive]
            k = min(8, len(pa))
            dist, nn = cKDTree(pa).query(cand, k=k)
            dist, nn = dist.reshape(len(cand), k), nn.reshape(len(cand), k)
            free = (dist > 0.5 * (cr[:, None] + ra[nn])).all(axis=1)
            cand, cr = cand[free], cr[free]
        new = []
        if len(cand):
            perm = self.rng.permutation(len(cand))
            cand, cr = cand[perm], cr[perm]
            nb = cKDTree(cand).query_ball_point(cand, r=cr * 1.5)
            free = np.ones(len(cand), bool)
            for i, neigh in enumerate(nb):
                if free[i]:
                    new.append(i)
                    neigh = np.asarray(neigh, int)
                    free[neigh[np.hypot(*(cand[neigh] - cand[i]).T) < 0.5 * (cr[neigh] + cr[i])]] = False
        new = np.array(new, int)
        self.stats['born'] = len(new)
        if len(new):
            npos = cand[new]
            good, lay, r, phi0 = self.label(grid, layer, Lx, Ly, t_ref, J, npos)
            m = good.sum()
            add = dict(lay=lay[good], r=r[good], phi0=phi0[good], born=np.full(m, f), rank=self.rng.random(m),
                       size=self.p.dot_size * self.rng.lognormal(0, self.p.dot_size_jitter, m),
                       ang=self.rng.uniform(0, 2*np.pi, m),
                       off=self.rng.uniform(self.p.lobe_offset_min, self.p.lobe_offset_max, m),
                       dying=np.zeros(m, int))
            for k in self.FIELDS:
                setattr(self, k, np.concatenate([getattr(self, k), add[k]]))
            pos = np.vstack([pos, npos[good]])
        self.stats['alive'] = int((self.dying == 0).sum())
        # size: growing in after birth, shrinking while fading out
        grow = np.minimum(1.0, (f - self.born + 1) / FADE)
        shrink = np.where(self.dying > 0, self.dying / FADE, 1.0)
        draw = (pos, self.size * grow * shrink, self.ang, self.off)
        self.dying = np.where(self.dying > 0, self.dying - 1, 0)
        self._take(~((self.dying == 0) & (shrink < 1)))  # drop dots that finished fading out
        return draw


def sample(field, xs, ys, pts):
    j = (pts[:, 0] - xs[0]) / (xs[1] - xs[0]); i = (pts[:, 1] - ys[0]) / (ys[1] - ys[0])
    return ndimage.map_coordinates(field, [i, j], order=1, mode='nearest')


# --- Drawing a frame -------------------------------------------------------------------
def ink_coverage(pos, size, ang, off, p, W, H, dpi, S=4):
    """Antialiased ink coverage of all dots (each a blob plus an offset lobe), as in the stills."""
    px_pt = dpi / 72
    pt_to_M = 2*VIEW / (bh.FIG_W * 72)
    view_h = VIEW / p.aspect
    lobe = pos + np.column_stack([np.cos(ang), np.sin(ang)]) * (off * size * pt_to_M)[:, None]
    def to_px(xy):
        return (view_h - xy[:, 1]) / (2*view_h) * H, (xy[:, 0] + VIEW) / (2*VIEW) * W
    cy, cx = to_px(pos); ly, lx = to_px(lobe)
    r1 = 0.5*size*px_pt; r2 = 0.5*p.lobe_scale*size*px_pt
    K = int(np.ceil(2*(max(r1.max(initial=0), 0.5) + np.hypot(cy - ly, cx - lx).max(initial=0)))) + 3
    P = K + 1
    T = np.ones((H + 2*P, W + 2*P), np.float32)
    Tf = T.reshape(-1)
    sub = (np.arange(K)[:, None] + (np.arange(S) + 0.5)/S).ravel()
    offsets = (np.arange(K)[:, None]*(W + 2*P) + np.arange(K)[None, :]).ravel()
    for s in range(0, len(cy), 4000):
        sl = slice(s, s + 4000)
        r0 = np.floor(cy[sl] - K/2).astype(int) + 1; c0 = np.floor(cx[sl] - K/2).astype(int) + 1
        inside = (r0 > -K) & (r0 < H) & (c0 > -K) & (c0 < W)
        if not inside.any():
            continue
        r0, c0 = r0[inside], c0[inside]
        n = len(r0)
        def cov(yc, xc, R):
            dy = (r0[:, None] + sub[None, :] - yc[sl][inside][:, None])[:, :, None]
            dx = (c0[:, None] + sub[None, :] - xc[sl][inside][:, None])[:, None, :]
            hit = dy*dy + dx*dx <= (R[sl][inside]**2)[:, None, None]
            return hit.reshape(n, K, S, K, S).mean(axis=(2, 4), dtype=np.float32)
        a = 1 - (1 - cov(cy, cx, r1)) * (1 - cov(ly, lx, r2))
        idx = ((r0 + P)*(W + 2*P) + (c0 + P))[:, None] + offsets[None, :]
        np.multiply.at(Tf, idx.ravel(), (1 - a).ravel())
    return 1 - T[P:-P, P:-P]


# --- Main ---------------------------------------------------------------------------------
def main():
    from luminet_animation import ffmpeg_command, lower_priority
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='flyby.mp4')
    ap.add_argument('--periapsis', type=float, default=10.0, help='closest approach, in M')
    ap.add_argument('--orbit-tilt', type=float, default=45.0, help="angle between the orbit's plane and the disk, in degrees")
    ap.add_argument('--retrograde-orbit', action='store_true', help='orbit against the disk rotation')
    ap.add_argument('--start-radius', type=float, default=80.0, help='where the flight starts and ends, in M')
    ap.add_argument('--fov', type=float, default=60.0, help='horizontal field of view, in degrees')
    ap.add_argument('--duration', type=float, default=24.0, help='video length, in seconds')
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--pacing', type=float, default=1.0,
                    help='slow-motion near the hole: 0 = the camera clock runs evenly, 1 = in proportion to distance')
    ap.add_argument('--disk-rotation', type=float, default=1.0, help='speed of the disk rotation (1 = physical)')
    ap.add_argument('--trace-width', type=int, default=840, help='ray-traced grid width (height = width / aspect)')
    ap.add_argument('--frames', type=float, nargs='*', default=None,
                    help='instead of a video, save single frames as PNGs at these points of the flight (0 to 1)')
    ap.add_argument('--seconds', type=float, default=None, help='stop the video after this many seconds (preview)')
    ap.add_argument('--range', type=float, nargs=2, default=None, metavar=('START', 'END'),
                    help='only render this part of the flight, as fractions from 0 to 1 (preview)')
    ap.add_argument('--texture', type=float, default=0.5, help='paper texture strength (1 = as in the stills)')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--crf', type=float, default=12)
    ap.add_argument('--x264-preset', default='slow')
    ap.add_argument('--pix-fmt', default='yuv420p')
    ap.add_argument('--lossless', action='store_true')
    bh.add_param_args(ap)
    args = ap.parse_args()
    lower_priority()
    p = bh.params_from_args(args).resolved()
    p.validate()
    p = __import__('dataclasses').replace(p, view=VIEW)
    a = p.spin

    path = camera_path(a, args.periapsis, args.orbit_tilt, not args.retrograde_orbit, args.start_radius)
    tau = path['tau']
    n_frames = round(args.duration * args.fps)
    # Pacing: each frame advances the camera's clock in proportion to (r / periapsis)^pacing,
    # so the brief close encounter gets more of the video than the long approach.
    s = np.concatenate([[0], np.cumsum(np.diff(tau) / (path['r'][1:] / args.periapsis)**args.pacing)])
    taus = np.interp(np.linspace(0, s[-1], n_frames), s, tau)
    print(f"orbit: L_z = {path['Lz']:.3f}, Q = {path['Q']:.3f}; {tau[-1] - tau[0]:.0f}M of proper time, "
          f"{n_frames} frames ({np.diff(taus).min():.2f}M to {np.diff(taus).max():.2f}M per frame)")

    Wt = args.trace_width; Ht = round(Wt / p.aspect)
    xs = np.linspace(-VIEW, VIEW, Wt); ys = np.linspace(-VIEW / p.aspect, VIEW / p.aspect, Ht)
    X, Y = np.meshgrid(xs, ys)
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    pool = ProcessPoolExecutor(workers)

    def rays_for(tau_f, X, Y):
        cam = camera_at(path, tau_f)
        v, gam, norm_err = camera_velocity(a, cam, path['Lz'])
        return cam, v, camera_rays(a, cam, v, X, Y, args.fov)

    # Exposure: a quick low-resolution pass over the flight, smoothed over time
    # (like a camera's auto exposure), so brightness doesn't flicker frame to frame.
    print("exposure pass...")
    xl = np.linspace(-VIEW, VIEW, 210); yl = np.linspace(-VIEW / p.aspect, VIEW / p.aspect, round(210 / p.aspect))
    XL, YL = np.meshgrid(xl, yl)
    sample_frames = np.unique(np.linspace(0, n_frames - 1, min(n_frames, 120)).round().astype(int))
    refs = []
    for fi in sample_frames:
        cam, v, rays = rays_for(taus[fi], XL, YL)
        traced = collect(*trace_frame(pool, a, rays, chunks=workers), len(rays['ell']))
        layer, R, PHI, TEM, F = frame_fields(p, cam, rays, traced, XL.shape)
        refs.append(exposure_reference(p, F, layer))
    log_ref = np.log(np.array(refs))
    log_ref = np.interp(np.arange(n_frames), sample_frames, np.where(np.isnan(log_ref), np.nanmedian(log_ref), log_ref))
    log_ref = ndimage.gaussian_filter1d(log_ref, 0.5 * args.fps, mode='nearest')
    ref = np.exp(log_ref)

    RES = 1
    W, H = round(bh.FIG_W * 200 * RES), round(bh.FIG_W / p.aspect * 200 * RES)
    tex = bh.Texturer(H, W, RES, p, strength=args.texture)
    blur = p.ink_blur * RES
    dots = Dots(p, p.seed, args.disk_rotation)

    if args.frames is not None:
        frame_list = [int(round(fr * (n_frames - 1))) for fr in args.frames]
        warm = 3 * FADE                                  # let the dots settle before each saved frame
    else:
        first, last = 0, n_frames
        if args.range:
            first, last = round(args.range[0] * (n_frames - 1)), round(args.range[1] * (n_frames - 1)) + 1
        if args.seconds is not None:
            last = min(last, first + round(args.seconds * args.fps))
        frame_list = list(range(first, last))
        warm = 0
    enc = None
    if args.frames is None:
        enc = subprocess.Popen(ffmpeg_command(args.out, W, H, args.fps, args), stdin=subprocess.PIPE)
    t0 = time.time()
    schedule = []
    for fi in frame_list:
        schedule += [(g, g == fi) for g in range(max(0, fi - warm), fi + 1)] if warm else [(fi, True)]
    pending = None
    for k, (fi, output) in enumerate(schedule):
        if pending is None:
            cam, v, rays = rays_for(taus[fi], X, Y)
            pending = (cam, rays, trace_frame(pool, a, rays))
        cam, rays, job = pending
        traced = collect(*job, len(rays['ell']))
        if k + 1 < len(schedule):                       # start tracing the next frame while this one is drawn
            ncam, nv, nrays = rays_for(taus[schedule[k + 1][0]], X, Y)
            pending = (ncam, nrays, trace_frame(pool, a, nrays))
        else:
            pending = None
        layer, R, PHI, TEM, F = frame_fields(p, cam, rays, traced, X.shape)
        dens = density_field(p, R, F, layer, ref[fi])
        pos, size, ang, off = dots.step(fi, (xs, ys), layer, R, PHI, TEM, dens, cam['t'])
        if not output:
            continue
        m = ink_coverage(pos, size, ang, off, p, W, H, 200 * RES)
        img = tex(ndimage.gaussian_filter(m, blur))
        if enc:
            enc.stdin.write(img.tobytes())
        else:
            from PIL import Image
            name = os.path.splitext(args.out)[0] + f"_{fi / (n_frames - 1):.3f}.png"
            Image.fromarray(img).save(name)
            print(f"  saved {name}")
        if fi % 60 == 0 or not enc or args.range:
            st = dots.stats
            print(f"  frame {fi}/{n_frames}: r = {cam['r']:.1f}M, {st['alive']} dots "
                  f"(+{st['born']} born, {st.get('faded', 0)} fading out, {st['lost']} out of view), "
                  f"{time.time() - t0:.0f} s", flush=True)
    if enc:
        enc.stdin.close()
        if enc.wait() != 0:
            sys.exit("ffmpeg failed")
    pool.shutdown()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
