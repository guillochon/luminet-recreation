"""
Stippled images of a black hole with a thin accretion disk, drawn the way
J.-P. Luminet inked his 1979 image.

Physics:
  1. Light bending. Non-spinning (Schwarzschild): the photon orbit equation
     u'' = -u + 3Mu^2, exactly as before. Spinning (Kerr): every pixel's ray is
     traced back through the Kerr geometry, with its radial and polar motion
     integrated in Mino time.
  2. Thin disk on circular orbits, from the innermost stable orbit outward, with
     the Page-Thorne (Novikov-Thorne, for spin) bolometric flux
  3. Gravitational + Doppler shift: F_obs = F / (1+z)^4
  4. Primary image + secondary ("ghost") image that wraps over the hole
  5. Rendered as dots, density ~ observed flux

Lengths are in units of the black hole's mass M (G = c = 1), so the mass itself
only sets the scale. Every setting lives in Params; its defaults are exactly the
values used to recreate Luminet's 1979 image (see luminet_1979.py).

    python blackhole.py --list-params
    python blackhole.py --set inclination=60 --out tilted.png
    python blackhole.py --preset edge-on --set approaching_side=right
    python blackhole.py --params my_hole.json
"""
import argparse
import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

M = 1.0


def _p(default, help):
    return field(default=default, metadata={'help': help})


@dataclass
class Params:
    # --- Black hole and disk ---
    inclination: float = _p(80.0, "observer's angle from the disk axis, in degrees (0 = face-on, 90 = edge-on)")
    spin: float = _p(0.0, "black hole spin a/M, from -0.998 to 0.998; positive spins the same way as the disk, "
                          "negative against it (0 = Schwarzschild, as Luminet)")
    r_in: float = _p(None, "disk inner edge, in M; default (isco) is the innermost stable orbit: 6M without spin, "
                           "down to 1.24M (prograde) or up to 9M (retrograde)")
    r_out: float = _p(400.0, "disk outer edge, in M")
    approaching_side: str = _p('left', "side of the disk rotating toward the observer: 'left' or 'right'")
    sharp_inner_edge: bool = _p(True, "hold intrinsic brightness at its peak inside r_peak, so the disk starts abruptly at r_in, as Luminet drew it")
    r_peak: float = _p(None, "radius of peak disk brightness, in M (used by sharp_inner_edge); default (auto) is "
                             "9.55M without spin, as in Luminet's paper, else the computed peak")
    edge_softness: float = _p(0.4, "with sharp_inner_edge: how far into the disk the fade-in from the shadow lasts, in M")
    luminet_ghost: bool = _p(False, "draw the secondary image the way Luminet did (a thin line at ghost_b) instead of computing it")
    ghost_b: float = _p(5.25, "with luminet_ghost: impact parameter of the ghost line, in M")
    ghost_width: float = _p(0.12, "with luminet_ghost: half-width of the ghost line, in M")
    # --- Frame ---
    view: float = _p(36.0, "half-width of the frame, in M")
    aspect: float = _p(2.0, "frame width / height")
    grid_w: int = _p(1680, "width of the internal brightness grid, in cells")
    n_b: int = _p(3500, "number of photon impact parameters traced")
    dphi: float = _p(0.002, "photon orbit integration step, in radians")
    # --- Stippling ---
    seed: int = _p(1979, "random seed for dot placement")
    banded: bool = _p(True, "ink more densely near a series of disk radii, like Luminet's isoradials")
    ring_step: float = _p(1.44, "with banded: each band's radius / the previous one's")
    band_depth: float = _p(0.10, "with banded: 0 = no banding, 1 = empty gaps between bands")
    auto_exposure: bool = _p(True, "brighten the image until at least as much of the disk is inked at full density as in "
                                   "Luminet's drawing (never darkens; matters for fast prograde spin, whose tiny, "
                                   "extremely bright inner disk would otherwise leave the rest nearly blank)")
    dot_gamma: float = _p(1.3, ">1 thins dim regions more than bright ones")
    dot_gain: float = _p(2.2, ">1 pushes mid-bright regions toward saturation")
    dot_cap: float = _p(0.96, "density ceiling: how packed the brightest region is")
    background_density: float = _p(0.0047, "faint speckle over the whole disk")
    dot_spacing: float = _p(0.05, "dot spacing in the brightest region, in M per 30 M of view")
    oversample: int = _p(8, "candidate dots per final dot")
    edge_rows: int = _p(4, "rows of evenly spaced dots along the disk's inner edge")
    edge_rows_within: float = _p(15.0, "only edges within this distance of the center get rows, in M")
    # --- Secondary ring, inked by hand: a dotted line plus a fan of dots ---
    ring_style: str = _p('auto', "secondary image: 'line' (Luminet's dotted line + fan), 'stipple' (dotted like the disk), "
                                 "or 'auto' (line unless the image is wider than ring_line_max_width, as when edge-on)")
    ring_line_max_width: float = _p(2.0, "with ring_style='auto': widest secondary image still drawn as a line, in M")
    ring_lopsided_power: float = _p(2.0, "Luminet's lopsided ring (fan on the bright side only, line spacing growing toward "
                                         "the dim side) is scaled by (ring's brightness asymmetry / Luminet's) ** this; "
                                         "0 = always fully lopsided")
    ring_spacing_bright: float = _p(0.17, "line dot spacing at the bright end, in M per 30 M of view")
    ring_spacing_dim: float = _p(0.34, "line dot spacing at the dim end, in M per 30 M of view")
    ring_second_row: int = _p(4, "dots in a short second row at the dim end")
    fan_unit: float = _p(0.19, "size unit for the fan, in M per 30 M of view")
    fan_max_width: float = _p(0.64 * 13 * 0.85, "fan width at its widest, in fan units")
    fan_taper: float = _p(0.6, "<1 makes the fan widen more gradually")
    fan_threshold: float = _p(0.12, "below this relative width x brightness, the ring stays 1 dot wide")
    fan_spacing: float = _p(0.78, "dot spacing in the fan, in fan units")
    fan_inside: float = _p(0.4, "fraction of the fan that spills inward over the line")
    # --- Ink and paper ---
    paper: str = _p('#221f1b', "paper color")
    ink: str = _p('#ece7cf', "ink color")
    dot_size: float = _p(1.5, "typical dot diameter, in points on a 14-inch-wide image")
    dot_size_jitter: float = _p(0.10, "log-normal spread of dot sizes")
    lobe_scale: float = _p(0.7, "each dot's offset lobe, relative to the dot's size")
    lobe_offset_min: float = _p(0.08, "smallest lobe offset, relative to dot size")
    lobe_offset_max: float = _p(0.25, "largest lobe offset, relative to dot size")
    ink_blur: float = _p(0.35, "edge softening, in pixels at 200 dpi")
    texture: bool = _p(True, "add paper grain, mottle and tonal drift")
    grain: float = _p(3.5, "paper grain strength, in 0-255 levels")
    mottle: float = _p(2.1, "mid-scale mottle strength, in 0-255 levels")
    drift: float = _p(2.8, "slow tonal drift strength, in 0-255 levels")
    edge_roughness: float = _p(0.07, "how much paper fiber eats into dot edges")

    @property
    def view_h(self):
        return self.view / self.aspect

    def resolved(self):
        """A copy with the spin-dependent defaults (r_in, r_peak) filled in."""
        r_in = isco(self.spin) if self.r_in is None else self.r_in
        if self.r_peak is not None:
            r_peak = self.r_peak
        elif self.spin == 0:
            r_peak = 9.55                      # Luminet's value: the Page-Thorne peak, to 3 figures
        else:
            r = np.linspace(isco(self.spin), 60.0, 200001)
            r_peak = float(r[np.nanargmax(novikov_thorne(r, self.spin))])
        return dataclasses.replace(self, r_in=r_in, r_peak=r_peak)

    def validate(self):
        if not 0 < self.inclination <= 90:
            raise ValueError("inclination must be in (0, 90] degrees")
        if not -0.998 <= self.spin <= 0.998:
            raise ValueError("spin must be in [-0.998, 0.998]")
        if self.r_in < isco(self.spin) - 1e-9:
            raise ValueError(f"r_in must be >= {isco(self.spin):.3f}: no stable disk inside the innermost stable orbit")
        if self.r_out <= self.r_in:
            raise ValueError("r_out must be larger than r_in")
        if self.approaching_side not in ('left', 'right'):
            raise ValueError("approaching_side must be 'left' or 'right'")
        if self.ring_style not in ('auto', 'line', 'stipple'):
            raise ValueError("ring_style must be 'auto', 'line' or 'stipple'")
        if self.view <= 0 or self.aspect <= 0:
            raise ValueError("view and aspect must be positive")


# Named starting points; each is a set of changes to the Luminet defaults.
PRESETS = {
    'luminet1979': {},
    'spinning': {'spin': 0.9},
    'retrograde': {'spin': -0.9},
    'max-spin-edge-on': {'spin': 0.998, 'inclination': 87.0, 'view': 30.0},
    # Seen face-on, Doppler beaming is weak; Luminet's contrast settings would
    # saturate the whole disk, so this preset spreads the brightness out.
    'face-on': {'inclination': 20.0, 'view': 30.0, 'dot_gain': 0.8, 'dot_gamma': 2.2, 'dot_cap': 0.8},
    'edge-on': {'inclination': 89.0},
    'truncated-disk': {'r_in': 14.0, 'r_peak': 14.0},
    'luminet-ghost': {'luminet_ghost': True},
}


# --- 1. Photon orbits ---------------------------------------------------------
def trace_photons(p):
    """Integrate photon orbits u(phi) = 1/r(phi) for a range of impact parameters b."""
    b_max = max(60.0, 1.1 * np.hypot(p.view, p.view_h))   # the frame's corners must be covered
    b_vals = np.linspace(0.02, b_max, p.n_b)
    dphi = p.dphi
    phis = np.arange(0, 2 * np.pi + 0.1, dphi)
    u = np.zeros_like(b_vals); du = 1.0 / b_vals
    U = np.zeros((len(phis), len(b_vals)))
    alive = np.ones_like(b_vals, bool); captured = np.zeros_like(alive)
    f = lambda u: -u + 3 * M * u**2
    for i in range(len(phis)):
        U[i] = np.where(alive, u, np.where(captured, 1.0, 0.0))  # captured -> inside hole, escaped -> infinity
        # RK4 step on (u, u')
        k1u, k1v = du, f(u)
        k2u, k2v = du + 0.5*dphi*k1v, f(u + 0.5*dphi*k1u)
        k3u, k3v = du + 0.5*dphi*k2v, f(u + 0.5*dphi*k2u)
        k4u, k4v = du + dphi*k3v, f(u + dphi*k3u)
        u = u + dphi/6*(k1u + 2*k2u + 2*k3u + k4u)
        du = du + dphi/6*(k1v + 2*k2v + 2*k3v + k4v)
        captured |= alive & (u >= 0.5/M)
        alive &= (u < 0.5/M) & (u > -1e-9)   # stop at horizon or after escaping
        u[~alive] = 0; du[~alive] = 0
    return RegularGridInterpolator((phis, b_vals), U, bounds_error=False, fill_value=np.nan)


def trace_kerr(p, X, Y, crossings=2, verbose=True, workers=None):
    """Trace each image-plane point (X, Y), in M, back from a distant observer through
    the Kerr geometry. Returns the radius of each ray's first `crossings` passes
    through the disk plane (NaN where it falls in or escapes first), and whether it
    fell into the hole. The rays are split across worker processes."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    shape = X.shape
    X, Y = X.ravel().astype(float), Y.ravel().astype(float)
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    if workers == 1 or len(X) < 50_000:
        out, fell = _trace_kerr_rays(p.spin, p.inclination, X, Y, crossings)
    else:
        if verbose: print(f"  tracing {len(X)} rays through the Kerr geometry on {workers} processes...", flush=True)
        chunks = np.array_split(np.arange(len(X)), workers * 4)
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_trace_kerr_rays, [p.spin] * len(chunks), [p.inclination] * len(chunks),
                                [X[c] for c in chunks], [Y[c] for c in chunks], [crossings] * len(chunks)))
        out = np.concatenate([o for o, _ in parts], axis=1)
        fell = np.concatenate([f for _, f in parts])
    return [o.reshape(shape) for o in out], fell.reshape(shape)


def _trace_kerr_rays(a, inclination, X, Y, crossings):
    """Kerr ray tracing for one batch of rays (see trace_kerr).

    In Mino time lambda the radial and polar motions decouple. With x = 1/r and
    u = cos(theta):
        (dx/dl)^2 = P(x) = (1 + (a^2 - a ell) x^2)^2 - x^2 (1 - 2x + a^2 x^2) (eta + (ell - a)^2)
        (du/dl)^2 = U(u) = eta + (a^2 - eta - ell^2) u^2 - a^2 u^4
    where ell and eta (Carter's constant) follow from the image position. Both
    are integrated in second-order form, x'' = P'(x)/2 and u'' = U'(u)/2, which
    passes through turning points smoothly. With a = 0 this is u'' = -u + 3u^2."""
    th = np.radians(min(inclination, 89.999))                # exactly edge-on, rays would start in the disk plane
    ell = -X * np.sin(th)
    eta = Y**2 + (X**2 - a*a) * np.cos(th)**2
    A = a*a - a*ell
    Q = eta + (ell - a)**2
    C = a*a - eta - ell*ell
    n = len(X)
    out = np.full((crossings, n), np.nan)
    # state, for the rays still being traced
    idx = np.arange(n)
    x = np.zeros(n); vx = np.ones(n)                         # x = 1/r = 0 at the observer, moving inward
    u = np.full(n, np.cos(th)); vu = Y * np.sin(th)          # rays above the image center head toward the pole first
    h = DPHI_KERR / np.maximum(np.sqrt(np.maximum(Q, 0)), 1.0)   # ~ constant step in orbital angle
    hits = np.zeros(n, int)
    fell = np.zeros(n, bool)
    x_horizon = 1 / (1 + np.sqrt(1 - a*a))
    def acc(x, u, A, Q, C):
        return (2*A*x*(1 + A*x*x) - Q*x*(1 - 3*x + 2*a*a*x*x),  C*u - 2*a*a*u**3)
    for step in range(MAX_STEPS_KERR):
        if step % 20 == 0:                                   # drop finished rays
            fell[idx[x >= x_horizon]] = True
            live = (hits < crossings) & (x < x_horizon) & ~((x < 0) & (vx < 0))
            if not live.all():
                idx, x, vx, u, vu, h, A, Q, C, hits = (v[live] for v in (idx, x, vx, u, vu, h, A, Q, C, hits))
            if len(idx) == 0:
                break
        # RK4 on (x, x', u, u')
        k1x, k1u = acc(x, u, A, Q, C)
        k2x, k2u = acc(x + 0.5*h*vx, u + 0.5*h*vu, A, Q, C)
        x2v, u2v = vx + 0.5*h*k1x, vu + 0.5*h*k1u
        k3x, k3u = acc(x + 0.5*h*x2v, u + 0.5*h*u2v, A, Q, C)
        x3v, u3v = vx + 0.5*h*k2x, vu + 0.5*h*k2u
        k4x, k4u = acc(x + h*x3v, u + h*u3v, A, Q, C)
        x4v, u4v = vx + h*k3x, vu + h*k3u
        x_new = x + h/6*(vx + 2*x2v + 2*x3v + x4v)
        u_new = u + h/6*(vu + 2*u2v + 2*u3v + u4v)
        vx = vx + h/6*(k1x + 2*k2x + 2*k3x + k4x)
        vu = vu + h/6*(k1u + 2*k2u + 2*k3u + k4u)
        # crossing the disk plane (u = 0) outside the horizon: interpolate x there
        crossed = (u > 0) != (u_new > 0)
        crossed &= (x > 0) & (x < x_horizon) & (hits < crossings)
        if crossed.any():
            i = np.nonzero(crossed)[0]
            f = u[i] / (u[i] - u_new[i])
            out[hits[i], idx[i]] = 1 / (x[i] + f * (x_new[i] - x[i]))
            hits[i] += 1
        x, u = x_new, u_new
    fell[idx[x >= x_horizon]] = True
    fell[idx[(hits < crossings) & (x > 0) & (x < x_horizon)]] = True   # still orbiting at the step limit
    return out, fell


DPHI_KERR = 0.01          # Kerr integration step, in radians of orbital angle (RK4)
MAX_STEPS_KERR = 1400     # rays still orbiting after ~14 radians count as falling in


# --- 2. Disk physics ------------------------------------------------------------
def page_thorne(r):
    # Luminet's Eq. 15 prints sqrt(3)/3 on the log term; sqrt(3)/2 is the correct
    # Page-Thorne value and is what reproduces the paper's own numbers (peak at
    # r=9.55M, 1.146e-4). The disk's stress vanishes at the innermost stable orbit, 6M.
    x = r / M
    s3, s6 = np.sqrt(3), np.sqrt(6)
    return (3*M/(8*np.pi)) / ((x - 3) * x**2.5) * (
        np.sqrt(x) - s6 + (s3/2) * np.log((np.sqrt(x)+s3)*(s6-s3) / ((np.sqrt(x)-s3)*(s6+s3))))


def isco(a):
    """Innermost stable circular orbit for spin a (a < 0: disk orbits against the spin)."""
    z1 = 1 + (1 - a*a)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
    z2 = np.sqrt(3*a*a + z1*z1)
    return float(3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2*z2)))


def novikov_thorne(r, a):
    """Page-Thorne flux for spin a (Page & Thorne 1974, Eq. 15n). It reduces to
    page_thorne() at a = 0, where it is numerically degenerate, so spin 0 uses that."""
    x = np.sqrt(r / M); x0 = np.sqrt(isco(a))
    t = np.arccos(a)
    x1, x2, x3 = 2*np.cos((t - np.pi)/3), 2*np.cos((t + np.pi)/3), -2*np.cos(t/3)   # roots of x^3 - 3x + 2a
    with np.errstate(divide='ignore', invalid='ignore'):
        term = x - x0 - 1.5*a*np.log(x/x0)
        for xi, xj, xk in ((x1, x2, x3), (x2, x1, x3), (x3, x1, x2)):
            term = term - 3*(xi - a)**2 / (xi*(xi - xj)*(xi - xk)) * np.log((x - xi) / (x0 - xi))
        return (3*M/(8*np.pi)) / (x**4 * (x**3 - 3*x + 2*a)) * term


def one_plus_z_kerr(r, ell, a):
    """Redshift of light with angular momentum ell (per unit energy) from a circular
    equatorial orbit at r: 1+z = u^t (1 - Omega ell)."""
    om = 1 / (r**1.5 + a)                                      # Keplerian angular velocity
    ut = 1 / np.sqrt(1 - 2/r + 4*a*om/r - (r*r + a*a + 2*a*a/r) * om*om)
    return ut * (1 - om * ell)


def observed_flux(p, r, b, alpha):
    incl = np.radians(p.inclination)
    if p.spin != 0:
        ell = -b * np.sin(alpha) * np.sin(incl)              # image x = b sin(alpha); ell = -x sin(incl)
        one_plus_z = one_plus_z_kerr(r, ell, p.spin)
        if p.sharp_inner_edge:
            fade = 1 - np.exp(-np.maximum(r - p.r_in, 0) / p.edge_softness)
            return fade * novikov_thorne(np.maximum(r, p.r_peak), p.spin) / one_plus_z**4
        return novikov_thorne(r, p.spin) / one_plus_z**4
    one_plus_z = (1 + np.sqrt(M / r**3) * b * np.sin(incl) * np.sin(alpha)) / np.sqrt(1 - 3*M/r)
    if p.sharp_inner_edge:
        # Luminet's drawing goes from dark shadow to dense dots abruptly, while
        # Page-Thorne brightness falls to zero at 6M: hold the intrinsic brightness
        # at its peak inside r_peak (gravitational redshift still dims it inward).
        r_src = np.maximum(r, p.r_peak)
        fade = 1 - np.exp(-np.maximum(r - p.r_in, 0) / p.edge_softness)
        return fade * page_thorne(r_src) / one_plus_z**4
    return page_thorne(r) / one_plus_z**4


# --- 3. Map every image-plane cell onto the disk ---------------------------------
def disk_image(p, interp):
    """Observed brightness and disk radius over the frame (approaching side on the left)."""
    VIEW, VIEW_H = p.view, p.view_h
    W = p.grid_w; H = int(round(W / p.aspect))
    incl = np.radians(p.inclination)
    xs = np.linspace(-VIEW, VIEW, W); ys = np.linspace(-VIEW_H, VIEW_H, H)
    X, Y = np.meshgrid(xs, ys)
    B = np.hypot(X, Y)
    ALPHA = np.arctan2(X, -Y)                      # alpha = 0 points at the near side of the disk
    cosA = np.cos(ALPHA)
    gamma = np.arccos(cosA / np.sqrt(cosA**2 + 1/np.tan(incl)**2))

    flux = np.zeros_like(B)
    RMAP = np.full_like(B, np.nan)   # disk radius seen at each cell
    # Secondary image. Luminet did not compute it fully: he drew it as a thin line at
    # b = 5.25M and overestimated its brightness with r = 9.55M, b = 5.25M in Eqs. 15 & 19.
    if p.spin != 0:
        r_kerr, _ = trace_kerr(p, X, Y)
    for n in ((0,) if p.luminet_ghost else (1, 0)):   # secondary first, primary overwrites (opaque disk)
        with np.errstate(divide='ignore', invalid='ignore'):
            if p.spin != 0:
                r = r_kerr[n]
            else:
                phi = gamma + n * np.pi
                un = interp(np.stack([phi.ravel(), B.ravel()], -1)).reshape(B.shape)
                r = 1.0 / un
            hit = (r >= p.r_in) & (r <= p.r_out)
            F = np.where(hit, observed_flux(p, np.where(hit, r, 10), B, ALPHA), 0)
        flux = np.where(hit, F, flux)
        RMAP = np.where(hit, r, RMAP)
        if n == 1: SEC = hit.copy()
        else: PRIM = hit
    SEC = (SEC & ~PRIM) if not p.luminet_ghost else np.zeros_like(PRIM)
    if p.luminet_ghost:
        SEC = (np.abs(B - p.ghost_b) < p.ghost_width) & np.isnan(RMAP)
        ghost = (np.abs(B - p.ghost_b) < p.ghost_width) & np.isnan(RMAP)
        flux = np.where(ghost, observed_flux(p, p.r_peak, p.ghost_b, ALPHA), flux)
    return dict(xs=xs, ys=ys, W=W, H=H, B=B, ALPHA=ALPHA, flux=flux, RMAP=RMAP, PRIM=PRIM, SEC=SEC)


# --- 4. Stipple it, like Luminet's India-ink dots ------------------------------------
# The secondary ring's brightness asymmetry as seen in the 1979 view (80 degrees),
# slightly rounded down so that view always counts as fully lopsided.
LUMINET_RING_ASYMMETRY = 0.411        # measured: 0.41187
# Fraction of the disk inked at full density in the 1979 view, slightly rounded
# down so that auto_exposure leaves that view exactly as it is.
LUMINET_SATURATED_FRACTION = 0.0105   # measured: 0.010617


def ring_asymmetry(ca, fb):
    """(bright left - dim right) / (sum) of the ring's mean brightness."""
    left = np.sin(ca) < 0
    if not left.any() or left.all():
        return 0.0
    l, r = fb[left].mean(), fb[~left].mean()
    return (l - r) / (l + r)


def stipple(p, img, verbose=True):
    from skimage import measure
    VIEW, VIEW_H = p.view, p.view_h
    xs, ys, W, H = img['xs'], img['ys'], img['W'], img['H']
    B, ALPHA, flux, RMAP, PRIM, SEC = (img[k] for k in ('B', 'ALPHA', 'flux', 'RMAP', 'PRIM', 'SEC'))
    rng = np.random.default_rng(p.seed)
    # Draw the secondary image as a hand-inked line only while it is thin enough to
    # read as one; seen nearly edge-on it becomes a broad arc, dotted like the disk.
    ring_style = p.ring_style
    if ring_style == 'auto':
        ring_style = 'line'
        if SEC.any():
            a_ = np.mod(ALPHA[SEC], 2*np.pi); b_ = B[SEC]
            k_ = np.digitize(a_, np.linspace(0, 2*np.pi, 721))
            widths = [np.ptp(b_[k_ == k]) for k in np.unique(k_)]
            if max(widths) > p.ring_line_max_width:
                ring_style = 'stipple'
    if ring_style == 'stipple':
        SEC = np.zeros_like(SEC)
    # Luminet plotted his dots along isoradials: curves of constant disk radius.
    # Mimic that by only inking near a discrete set of radii, boosting those
    # bands so the average dot density still follows the observed flux.
    weight = np.ones_like(B)
    if p.banded:
        k = np.log(RMAP / p.r_in) / np.log(p.ring_step)          # fractional ring index
        ring = 1 + p.band_depth * np.cos(2 * np.pi * np.nan_to_num(k))   # peaks on each ring
        weight = np.where(np.isnan(RMAP), 1.0, ring)
    # Brightness reference: the brightest point, as Luminet did, unless that would
    # leave less of the disk at full density than in his drawing (auto_exposure).
    ref = np.nanmax(flux)
    disk_flux = flux[~np.isnan(RMAP)]
    if p.auto_exposure and disk_flux.size:
        full = (p.dot_cap / p.dot_gain) ** (1 / p.dot_gamma)        # brightness, relative to ref, that reaches dot_cap
        ref_auto = np.quantile(disk_flux, 1 - LUMINET_SATURATED_FRACTION) / full
        if ref_auto < ref:
            if verbose: print(f"auto exposure: brightened {ref / ref_auto:.1f}x")
            ref = ref_auto
    disk_density = np.minimum(p.dot_gain * np.nan_to_num((flux / ref) ** p.dot_gamma), p.dot_cap) * weight
    disk_density = np.where(SEC, 0.0, disk_density)   # the ring band is inked separately, in rows
    density = disk_density + np.where(np.isnan(RMAP), 0.0, p.background_density)   # on the disk only, none in the hole

    # Hand-inked look: Luminet's dots are evenly spaced, not random, and a bit blobby.
    # 1) Draw many candidate dots in proportion to brightness.
    # 2) Poisson-disk elimination: walk through them in random order and keep a dot only
    #    if no kept dot lies within its spacing radius. The radius shrinks where it's
    #    brighter (spacing ~ 1/sqrt(density)), so brightness becomes dot spacing.
    SPACING_AT_PEAK = p.dot_spacing * VIEW / 30         # dot spacing in the brightest region, in units of M
    density = np.minimum(density, 1.0); dens_max = 1.0   # boosted ring is capped, not used to normalize
    # expected final dots ~ (brightness-weighted area) / spacing^2
    weighted_area = (density / dens_max).mean() * (2*VIEW) * (2*VIEW_H)
    n_cand = int(p.oversample * weighted_area / SPACING_AT_PEAK**2)

    # Regular rows along the disk's inner edge, the way Luminet kept that edge clean:
    # row 0 follows the edge with even spacing; each further row steps into the disk and
    # gets more positional jitter, so by edge_rows rows in it blends into random placement.
    EDGE_ROWS = p.edge_rows
    px_M = 2 * VIEW / (W - 1)
    dist = ndimage.distance_transform_edt(PRIM) * px_M            # distance into the disk from its edge, in M
    dens_inner = ndimage.maximum_filter(density, size=max(3, int(1.0 / px_M)))
    s_rows = SPACING_AT_PEAK / np.sqrt(np.maximum(dens_inner, 1e-4))   # row-to-row gap (ignores the edge fade)
    s_true = SPACING_AT_PEAK / np.sqrt(np.maximum(density, 1e-4))      # local dot spacing (includes the fade)
    u = dist / s_rows
    def at(field, x, y):
        j = np.clip(((x + VIEW) / (2*VIEW) * (W - 1)).round().astype(int), 0, W - 1)
        i = np.clip(((y + VIEW_H) / (2*VIEW_H) * (H - 1)).round().astype(int), 0, H - 1)
        return field[i, j]
    edge_dots = []
    for k in range(EDGE_ROWS):
        for c in measure.find_contours(u, 0.5 + 0.9 * k):
            cxs = np.interp(c[:, 1], np.arange(W), xs); cys = np.interp(c[:, 0], np.arange(H), ys)
            if np.hypot(cxs, cys).min() > p.edge_rows_within: continue   # only the edge around the hole
            sl = np.concatenate([[0], np.cumsum(np.hypot(np.diff(cxs), np.diff(cys)))])
            t = rng.uniform(0, 0.5) * at(s_true, cxs[:1], cys[:1])[0]
            while t < sl[-1]:
                x0, y0 = np.interp(t, sl, cxs), np.interp(t, sl, cys)
                step = min(at(s_true, np.array([x0]), np.array([y0]))[0],
                           3 * at(s_rows, np.array([x0]), np.array([y0]))[0])
                jit = (k / EDGE_ROWS) * 0.45 * step
                edge_dots.append((x0 + rng.normal(0, jit), y0 + rng.normal(0, jit)))
                t += step
    edge_dots = np.array(edge_dots).reshape(-1, 2)

    cand = []
    while sum(len(c) for c in cand) < n_cand:
        n = 4_000_000
        cx = rng.uniform(-VIEW, VIEW, n); cy = rng.uniform(-VIEW_H, VIEW_H, n)
        ix = np.clip(((cx + VIEW) / (2*VIEW) * (W - 1)).round().astype(int), 0, W - 1)
        iy = np.clip(((cy + VIEW_H) / (2*VIEW_H) * (H - 1)).round().astype(int), 0, H - 1)
        d = density[iy, ix] / dens_max
        ok = rng.random(n) < d
        cand.append(np.column_stack([cx[ok], cy[ok], d[ok]]))
    cand = np.concatenate(cand)
    pts, dloc = cand[:, :2], cand[:, 2]
    radius = SPACING_AT_PEAK / np.sqrt(dloc)
    tree = cKDTree(pts)
    order = rng.permutation(len(pts))
    alive = np.ones(len(pts), bool); kept = []
    if len(edge_dots):                                             # edge rows are placed first; keep clear of them
        alive &= cKDTree(edge_dots).query(pts)[0] > 0.9 * radius
    neighbors = tree.query_ball_point(pts, r=radius)
    for i in order:
        if alive[i]:
            kept.append(i)
            alive[neighbors[i]] = False
    kept = np.array(kept, int)
    dots = np.vstack([edge_dots, pts[kept]])
    if verbose: print("edge-row dots:", len(edge_dots))

    # The secondary ring is inked separately, the way Luminet did it: a carefully dotted
    # line along the ring's inner edge, plus a fan of evenly spaced (but not gridded) dots
    # behind it where the ring broadens. The fan's width follows ring width x brightness,
    # so it flares out on the bright lower left, barely appears on the lower right, and
    # its dots are spaced together with the disk's so the fan merges into the disk.
    RING_SPACING_LEFT = p.ring_spacing_bright * VIEW / 30   # distance between line dots at the left (bright) end, in M
    RING_SPACING_RIGHT = p.ring_spacing_dim * VIEW / 30     # ...and at the right (dim) end
    RIGHT_FAN_DOTS = p.ring_second_row          # second row of dots along the last few line dots on the right
    FAN_UNIT = p.fan_unit * VIEW / 30           # fan sizes are set in this unit, independent of the line spacing
    FAN_MAX_WIDTH = p.fan_max_width * FAN_UNIT  # fan width at its widest, where it meets the disk
    FAN_SPACING = p.fan_spacing * FAN_UNIT      # dot spacing inside the fan
    sy, sx = np.nonzero(SEC)
    if len(sx):
        a = np.mod(ALPHA[sy, sx], 2*np.pi); bb = B[sy, sx]   # 0..2pi so the ring runs unbroken over the top
        bins = np.linspace(0, 2*np.pi, 721)
        idx = np.digitize(a, bins)
        ks = np.unique(idx)
        ca = np.array([a[idx == k].mean() for k in ks])
        bmin = np.array([bb[idx == k].min() for k in ks])
        bmax = np.array([bb[idx == k].max() for k in ks])
        fsec = flux[sy, sx]
        fb = np.array([fsec[idx == k].mean() for k in ks])
        sm = lambda v: np.convolve(np.pad(v, 3, mode='edge'), np.ones(7) / 7, mode='valid')
        bmin, bmax, fb = sm(bmin), sm(bmax), sm(fb)       # smooth out pixel stair-steps
        weight_w = (bmax - bmin) * fb
        w_rel = np.clip((weight_w / max(weight_w.max(), 1e-300) - p.fan_threshold) / (1 - p.fan_threshold), 0, 1)
        fan_w = FAN_MAX_WIDTH * w_rel ** p.fan_taper
        # Luminet drew the ring lopsided: fan on the bright left only, line dots
        # thinning toward the right. That matches the ring's Doppler asymmetry seen
        # from his 80 degrees. Elsewhere, scale the lopsidedness by how asymmetric
        # the ring really is; at or above Luminet's asymmetry, draw it exactly his way.
        left = np.sin(ca) < 0
        asym = ring_asymmetry(ca, fb)
        lop =min(1.0, max(asym, 0.0) / LUMINET_RING_ASYMMETRY) ** p.ring_lopsided_power
        if lop >= 1.0:
            fan_w = np.where(left, fan_w, 0.0)            # fan only on the left; the right stays a clean single line
        else:
            # fade the fan out across the top and bottom instead of cutting it at 12 and 6 o'clock,
            # and keep (1 - lop) of it on the dim side
            to_left = np.clip(0.5 - np.sin(ca) / 0.6, 0, 1)
            fan_w = fan_w * (lop * to_left + (1 - lop))
            mid = 0.5 * (RING_SPACING_LEFT + RING_SPACING_RIGHT)
            RING_SPACING_LEFT = mid + (RING_SPACING_LEFT - mid) * lop
            RING_SPACING_RIGHT = mid + (RING_SPACING_RIGHT - mid) * lop
            RIGHT_FAN_DOTS = int(round(RIGHT_FAN_DOTS * lop))
        if verbose and lop < 1.0: print(f"ring asymmetry {asym:.3f} -> lopsidedness {lop:.2f}")
        gapped = np.concatenate([[False], np.diff(ca) > np.radians(3)])
        contiguous = ~(gapped | np.roll(gapped, -1))       # break on both sides of a gap
        # 1) the carefully dotted inner line
        rx, ry = bmin * np.sin(ca), -bmin * np.cos(ca)
        # spacing grows steadily from the left end to the right end (ca runs right -> left)
        keep_pts = contiguous.copy()
        lx, ly = rx[keep_pts], ry[keep_pts]
        sl = np.concatenate([[0], np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))])
        L = sl[-1]; t = [0.0]
        while L > 0:
            f = t[-1] / L                                   # 0 at the right end, 1 at the left end
            step = RING_SPACING_RIGHT + (RING_SPACING_LEFT - RING_SPACING_RIGHT) * f
            if t[-1] + step > L + 1e-9: break
            t.append(t[-1] + step)
        t = np.array(t)
        line = np.column_stack([np.interp(t, sl, lx), np.interp(t, sl, ly)])
        # right end: a short, carefully placed second row just outside the line,
        # staggered between the last few line dots, so the ring is 2 dots wide there
        first = line[:RIGHT_FAN_DOTS + 1]
        mids = 0.5 * (first[1:] + first[:-1])
        outward = mids / np.linalg.norm(mids, axis=1, keepdims=True)
        gaps = np.linalg.norm(np.diff(first, axis=0), axis=1)
        line = np.vstack([line, mids + outward * (0.75 * gaps)[:, None]])
        d_line, _ = cKDTree(line).query(dots)
        dots = dots[d_line > 0.8 * RING_SPACING_LEFT]
        # 2) the fan: candidates spread evenly over the fan's area, then the same
        #    minimum-distance rule, checked against the line and existing disk dots too
        # the fan straddles the line: part of it lies inside, over the inner circle
        b_in = bmin - p.fan_inside * fan_w
        span = fan_w
        dA = span * bmin * np.radians(0.5)                 # area of each 0.5-degree slice
        fan = []
        if dA.sum() > 0:
            n_c = int(30 * dA.sum() / FAN_SPACING**2) + 1
            pick = rng.choice(len(ca), n_c, p=dA / dA.sum())
            c_a = ca[pick] + rng.uniform(-0.5, 0.5, n_c) * np.radians(0.5)
            c_b = b_in[pick] + rng.uniform(0, 1, n_c) * span[pick]
            cand_f = np.column_stack([c_b * np.sin(c_a), -c_b * np.cos(c_a)])
            fixed = np.vstack([dots, line]); fixed_tree = cKDTree(fixed)
            ok = fixed_tree.query(cand_f)[0] > FAN_SPACING
            cand_f = cand_f[ok]
            for q in cand_f[rng.permutation(len(cand_f))]:
                if not fan or np.min(np.hypot(*(np.array(fan) - q).T)) > FAN_SPACING:
                    fan.append(q)
        fan = np.array(fan).reshape(-1, 2)
        dots = np.vstack([dots, line, fan])
        if verbose: print("ring line dots:", len(line), " fan dots:", len(fan))
    return dots, rng


# --- 5. Ink dots ---------------------------------------------------------------------
FIG_W = 14                                             # figure width in inches (2 * view in M)


@dataclass
class Drawing:
    """The finished dot layout: dot centers, sizes (points) and lobe centers, in M."""
    p: Params
    dots: np.ndarray
    size: np.ndarray
    lobe: np.ndarray

    @property
    def fig_size(self):
        return FIG_W, FIG_W / self.p.aspect

    def pixels(self, res_scale):
        dpi = 200 * res_scale
        return round(FIG_W * dpi), round(FIG_W / self.p.aspect * dpi)


def make_drawing(p=None, verbose=True):
    """Trace light, compute the disk's brightness and place every ink dot."""
    p = (p or Params()).resolved()
    p.validate()
    img = disk_image(p, trace_photons(p) if p.spin == 0 else None)
    dots, rng = stipple(p, img, verbose)
    # Blobby ink dots: each dot is a main blob plus a small offset lobe, with
    # size varying from dot to dot.
    size = p.dot_size * rng.lognormal(0, p.dot_size_jitter, len(dots))
    ang = rng.uniform(0, 2 * np.pi, len(dots)); off = rng.uniform(p.lobe_offset_min, p.lobe_offset_max, len(dots))
    pt_to_M = 2*p.view / (FIG_W * 72)                       # figure is 14 in wide for 2*view M
    lobe = dots + np.column_stack([np.cos(ang), np.sin(ang)]) * (off * size * pt_to_M)[:, None]
    if p.approaching_side == 'right':
        # everything above assumes the approaching side is on the left; the other
        # rotation direction is its exact mirror image
        dots = dots * [-1, 1]; lobe = lobe * [-1, 1]
    return Drawing(p, dots, size, lobe)


# --- 6. Paper and photographic texture --------------------------------------------
# The original isn't flat black: paper fibers, ink wicking and the photographic
# reproduction leave a fine grain, a patchy mid-scale mottle and slow tonal drift,
# all in a warm dark gray. Applied as a final pass; dot placement is untouched.
def hex_rgb(c):
    return np.array([int(c[i:i+2], 16) for i in (1, 3, 5)], float)


def texture_layers(h_, w_, res_scale):
    """Fixed noise fields (grain, mottle, drift) for an h_ x w_ image."""
    from PIL import Image
    trng = np.random.default_rng(7)
    def noise(sigma):
        # pixel sigmas are given at res_scale = 1; large ones are made at low resolution and upsampled
        sigma = sigma * res_scale
        f = max(1, int(sigma // 8))
        n = trng.normal(size=(h_ // f + 1, w_ // f + 1)).astype(np.float32)
        n = ndimage.gaussian_filter(n, sigma / f)
        if f > 1:
            n = np.asarray(Image.fromarray(n).resize((w_, h_), Image.BICUBIC))
        n = n[:h_, :w_]
        return (n - n.mean()) / n.std()
    grain = noise(0.8)                   # paper fibers / film grain
    mottle = noise(5)                    # patchy unevenness a few dots across
    drift = noise(140)                   # slow exposure variation across the print
    return grain, mottle, drift


class Texturer:
    """Turns an ink-coverage map m (0 = bare paper, 1 = solid dot) into the final RGB image."""
    def __init__(self, h_, w_, res_scale, p, strength=1.0):
        # strength scales every texture effect (grain, mottle, tonal drift, rough edges)
        if not p.texture:
            strength = 0.0
        grain, mottle, drift = texture_layers(h_, w_, res_scale)
        grain, mottle, drift = strength * grain, strength * mottle, strength * drift
        paper, ink = hex_rgb(p.paper), hex_rgb(p.ink)
        self.rough = p.edge_roughness * grain * 4
        self.paper = paper[None, None, :] + (p.grain * grain + p.mottle * mottle + p.drift * drift)[..., None]
        self.ink = ink[None, None, :] * (1 + 0.01 * grain + 0.02 * drift)[..., None]
    def __call__(self, m):
        return self.patch(m, (slice(None), slice(None)))
    def patch(self, m, region):
        """Texture just the pixels in region (a pair of slices); m covers that region."""
        # rough, fibrous dot edges: perturb the ink mask most where it is partial (edges)
        m = np.clip(m + self.rough[region] * m * (1 - m), 0, 1)
        out = self.paper[region] * (1 - m[..., None]) + self.ink[region] * m[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


def ink_coverage(img, p):
    """How much ink each pixel of an RGB image holds, 0 = bare paper, 1 = solid dot."""
    paper, ink = hex_rgb(p.paper), hex_rgb(p.ink)
    d = ink - paper
    return np.clip(((img - paper) @ d) / (d @ d), 0, 1)


def render_still(d, path, res_scale=2):
    """Render a Drawing to a PNG: 2800 x 1400 px per res_scale for a 2:1 frame."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageFilter
    p = d.p
    fig = plt.figure(figsize=d.fig_size, facecolor=p.paper)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(p.paper)
    ax.scatter(d.dots[:, 0], d.dots[:, 1], s=d.size**2, c=p.ink, linewidths=0)
    ax.scatter(d.lobe[:, 0], d.lobe[:, 1], s=(p.lobe_scale * d.size)**2, c=p.ink, linewidths=0)
    ax.set_xlim(-p.view, p.view); ax.set_ylim(-p.view_h, p.view_h); ax.axis('off')
    fig.savefig(path, dpi=200 * res_scale, facecolor=p.paper)
    plt.close(fig)
    # Soften edges slightly, like ink soaking into paper
    Image.open(path).filter(ImageFilter.GaussianBlur(p.ink_blur * res_scale)).save(path)
    if p.texture:
        img = np.asarray(Image.open(path).convert('RGB')).astype(np.float32)
        m = ink_coverage(img, p)
        Image.fromarray(Texturer(*m.shape, res_scale, p)(m)).save(path)


# --- Choosing parameters ------------------------------------------------------------
def add_param_args(ap):
    g = ap.add_argument_group('black hole parameters')
    g.add_argument('--preset', default='luminet1979', choices=sorted(PRESETS),
                   help='starting point (default: luminet1979, the exact 1979 recreation)')
    g.add_argument('--params', metavar='FILE', help='JSON file of parameter values, applied after the preset')
    g.add_argument('--set', metavar='NAME=VALUE', action='append', default=[],
                   help='set one parameter, e.g. --set inclination=60 (repeatable; applied last)')
    g.add_argument('--list-params', action='store_true', help='list every parameter with its default and exit')


def _convert(name, value):
    kind = {f.name: f.type for f in dataclasses.fields(Params)}.get(name)
    if kind is None:
        raise SystemExit(f"unknown parameter {name!r}; see --list-params")
    if kind in ('bool', bool):
        if isinstance(value, bool): return value
        return str(value).lower() in ('1', 'true', 'yes', 'on')
    if name in ('r_in', 'r_peak') and (value is None or str(value).lower() in ('isco', 'auto', 'none', '')):
        return None                                   # use the spin-dependent default
    return {'float': float, 'int': int, 'str': str}.get(kind, kind)(value)


def params_from_args(args):
    if args.list_params:
        for f in dataclasses.fields(Params):
            print(f"  {f.name:22s} {f.default!r:12}  {f.metadata['help']}")
        raise SystemExit
    values = dict(PRESETS[args.preset])
    if args.params:
        with open(args.params) as fh:
            values.update(json.load(fh))
    for item in args.set:
        name, _, value = item.partition('=')
        values[name.strip()] = value.strip()
    return Params(**{k: _convert(k, v) for k, v in values.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='blackhole.png', help='output PNG; the parameters are saved next to it as .json')
    ap.add_argument('--res-scale', type=float, default=2, help='resolution multiplier (1 = 2800 px wide)')
    add_param_args(ap)
    args = ap.parse_args()
    p = params_from_args(args)
    d = make_drawing(p)
    render_still(d, args.out, args.res_scale)
    with open(args.out.rsplit('.', 1)[0] + '.json', 'w') as fh:
        json.dump(dataclasses.asdict(p), fh, indent=2)
    print(f"{len(d.dots)} dots -> {args.out}")


if __name__ == '__main__':
    main()
