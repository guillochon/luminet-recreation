"""
Fly past the binary black hole of merger.py, filmed as a stippled ink drawing.

The camera falls in from far away on a free-fall (geodesic) path through the
binary's own spacetime, swings past the pair and leaves again, always aimed at
the binary's center of mass. Every frame is ray traced from the camera's actual
position and 4-velocity with merger.py's tracer, so the view includes the
camera's aberration and Doppler shift, the holes' orbital motion (they keep
moving while the light crosses them) and, for tilted spins, their precession.
The ink dots are attached to the disk as in flyby.py.

    python binary_flyby.py --set separation=5 --set phase=90 --set r_in=1 --set step=0.04 \\
        --set spin1=0.9 --set spin2=0.9 --set tilt1=90 --set tilt2=90 --set spin_az2=180
    ... --frames 0.5          # one frame at closest approach, as PNG
"""
import argparse
import dataclasses
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

import blackhole as bh
import flyby as fb
import merger as mg

VIEW = fb.VIEW
LAYERS = (0, 1, 2)        # dot layers: direct image, secondary, and all higher orders together
RIM = 1.0                 # ink density of the rim drawn along the shadows' edges (like the stills' inked outlines)
RIM_WIDTH = 0.35          # its e-folding width, in screen units


# --- Metric helpers ----------------------------------------------------------------
def lower(T, w_t, w_s):
    """g_ab w^b for g = eta + sum H l l: returns (w_t lowered, w_s lowered)."""
    lo_t, lo_s = -w_t, w_s.copy()
    for h in T:
        lw = h['lt'] * w_t + np.sum(h['ls'] * w_s, 0)
        lo_t = lo_t + h['H'] * h['lt'] * lw
        lo_s = lo_s + h['H'] * h['ls'] * lw
    return lo_t, lo_s


def dot(T, a, b):
    """g(a, b) for 4-vectors a = (a_t, a_s), b = (b_t, b_s)."""
    lt, ls = lower(T, a[0], a[1])
    return lt * b[0] + np.sum(ls * b[1], 0)


# --- The camera's path ---------------------------------------------------------------
def parabola_start(r_p, tilt_deg, r_far, prograde=True):
    """Position and velocity, at radius r_far on the way in, of a Newtonian parabolic orbit
    about the total mass with periapsis r_p, whose plane is tilted tilt_deg from the disk.
    The periapsis is the orbit's highest point above the disk, as in flyby.py."""
    i = np.radians(tilt_deg)
    P = np.array([0.0, np.cos(i), np.sin(i)])                # toward periapsis
    Vd = np.array([-1.0 if prograde else 1.0, 0.0, 0.0])      # direction of motion at periapsis
    nu = -np.arccos(2 * r_p / r_far - 1)                     # true anomaly on the way in
    x = r_far * (np.cos(nu) * P + np.sin(nu) * Vd)
    v = np.sqrt(1 / (2 * r_p)) * (-np.sin(nu) * P + (1 + np.cos(nu)) * Vd)
    D = np.tan(nu / 2)
    t_peri = np.sqrt(2 * r_p**3) * (D + D**3 / 3)            # Barker's equation: time from periapsis (negative)
    return x, v, t_peri


def geodesic(binary, x, t, u_t, u_s, r_stop, max_tau=5000.0, stop_at_periapsis=False):
    """Integrate a timelike geodesic forward in proper time (Hamiltonian form, p_t
    integrated) from contravariant 4-velocity u, until the radius passes r_stop on
    the way out. Returns arrays tau, t, x (3, n), u_t, u_s (3, n)."""
    T = mg.metric_terms(binary, np.array([t]), x[:, None])
    p_t, p_s = lower(T, np.array([u_t]), u_s[:, None])
    t = np.array([t]); x = x[:, None].astype(float)
    def rhs(t, x, ps, pt):
        T = mg.metric_terms(binary, t, x, dt=True)
        kt, ks = mg.raise_index(pt, ps, T)
        return kt, ks, mg.force(kt, ks, T), mg.force_t(kt, ks, T)
    rows = []; tau = 0.0; r_prev = np.inf; inbound = True
    while tau < max_tau:
        kt, ks, F, Ft = rhs(t, x, p_s, p_t)
        rows.append((tau, t[0], *x[:, 0], kt[0], *ks[:, 0]))
        r = float(np.sqrt(np.sum(x * x)))
        if r > r_prev:
            if inbound and stop_at_periapsis:
                break
            inbound = False
        if not inbound and r > r_stop:
            break
        r_prev = r
        h = 0.01 * min(r, 60.0)
        k2 = rhs(t + 0.5*h*kt, x + 0.5*h*ks, p_s + 0.5*h*F, p_t + 0.5*h*Ft)
        k3 = rhs(t + 0.5*h*k2[0], x + 0.5*h*k2[1], p_s + 0.5*h*k2[2], p_t + 0.5*h*k2[3])
        k4 = rhs(t + h*k3[0], x + h*k3[1], p_s + h*k3[2], p_t + h*k3[3])
        t = t + h/6 * (kt + 2*k2[0] + 2*k3[0] + k4[0])
        x = x + h/6 * (ks + 2*k2[1] + 2*k3[1] + k4[1])
        p_s = p_s + h/6 * (F + 2*k2[2] + 2*k3[2] + k4[2])
        p_t = p_t + h/6 * (Ft + 2*k2[3] + 2*k3[3] + k4[3])
        tau += h
    a = np.array(rows)
    return dict(tau=a[:, 0], t=a[:, 1], x=a[:, 2:5].T, u_t=a[:, 5], u_s=a[:, 6:9].T)


def camera_path(binary, r_p, tilt, r_far, prograde=True):
    """The camera's free fall: starts at r_far, moving as on a Newtonian parabola, timed
    so that it would pass periapsis at t = 0. The periapsis parameter is adjusted
    (secant method) so the real closest approach in the binary's spacetime is r_p."""
    def start(q):
        x, v, t0 = parabola_start(q, tilt, r_far, prograde)
        T = mg.metric_terms(binary, np.array([t0]), x[:, None], derivs=False)
        w = (np.array([1.0]), v[:, None])
        g = 1 / np.sqrt(-dot(T, w, w))[0]                    # normalize u = gamma (1, v)
        return x, t0, g, g * v
    def closest(q):
        x, t0, ut, us = start(q)
        path = geodesic(binary, x, t0, ut, us, r_far, stop_at_periapsis=True)
        return np.sqrt(np.sum(path['x']**2, 0)).min()
    q0, q1 = r_p, r_p * 1.05
    f0, f1 = closest(q0) - r_p, closest(q1) - r_p
    for _ in range(6):
        if abs(f1) < 1e-4:
            break
        q0, q1, f0 = q1, q1 - f1 * (q1 - q0) / (f1 - f0), f1
        f1 = closest(q1) - r_p
    x, t0, ut, us = start(q1)
    path = geodesic(binary, x, t0, ut, us, r_far)
    r = np.sqrt(np.sum(path['x']**2, 0))
    k = np.argmin(r)
    path.update(r=r, r_min=r[k], t_peri=path['t'][k], tau_peri=path['tau'][k], q=q1)
    return path


def camera_at(binary, path, tau):
    """Camera position, time and (re-normalized) 4-velocity at proper time tau."""
    t = np.interp(tau, path['tau'], path['t'])
    x = np.array([np.interp(tau, path['tau'], path['x'][i]) for i in range(3)])
    ut = np.interp(tau, path['tau'], path['u_t'])
    us = np.array([np.interp(tau, path['tau'], path['u_s'][i]) for i in range(3)])
    T = mg.metric_terms(binary, np.array([t]), x[:, None], derivs=False)
    n = np.sqrt(-dot(T, (np.array([ut]), us[:, None]), (np.array([ut]), us[:, None])))[0]
    return dict(t=t, x=x, u=(ut / n, us / n), r=float(np.linalg.norm(x)), T=T)


def camera_rays(cam, X, Y, fov_deg):
    """Backward rays for screen points (X, Y): starting positions, times and covariant
    momenta, for a camera with 4-velocity u aimed at the apparent position of the
    center of mass, with "up" toward the disk's north. The camera measures each
    photon's energy as 1."""
    T = cam['T']
    u = (np.array([cam['u'][0]]), cam['u'][1][:, None])
    def orth(v, basis):
        """Remove the components of v along u (timelike) and the unit spacelike basis vectors, then normalize."""
        c = dot(T, v, u)                                     # g(u, u) = -1
        v = (v[0] + c * u[0], v[1] + c * u[1])
        for b in basis:
            c = dot(T, v, b)
            v = (v[0] - c * b[0], v[1] - c * b[1])
        n = np.sqrt(dot(T, v, v))
        return (v[0] / n, v[1] / n)
    x = cam['x']
    # aim at where the center of mass appears: light from it arrives moving outward along x
    # (ignoring its bending), and the camera's motion aberrates that direction
    n_out = (x / np.linalg.norm(x))[:, None]
    gtt = dot(T, (np.ones(1), 0 * n_out), (np.ones(1), 0 * n_out))
    gts = dot(T, (np.ones(1), 0 * n_out), (np.zeros(1), n_out))
    gss = dot(T, (np.zeros(1), n_out), (np.zeros(1), n_out))
    kt = (-gts + np.sqrt(gts * gts - gtt * gss)) / gtt  # g(k, k) = 0 for k = (kt, n_out), future pointing
    kt = np.where(kt > 0, kt, (-gts - np.sqrt(gts * gts - gtt * gss)) / gtt)
    fwd = orth((-kt, -n_out), [])
    up = orth((np.zeros(1), np.array([[0.0], [0.0], [1.0]])), [fwd])
    right = orth((np.zeros(1), np.cross(fwd[1][:, 0], up[1][:, 0])[:, None]), [fwd, up])
    S = np.tan(np.radians(fov_deg) / 2) / VIEW
    a, b = (X.ravel() * S)[None], (Y.ravel() * S)[None]
    nrm = np.sqrt(1 + a*a + b*b)
    d_t = (fwd[0] + a * right[0] + b * up[0]) / nrm
    d_s = (fwd[1] + a * right[1] + b * up[1]) / nrm
    k_t = u[0] - d_t; k_s = u[1] - d_s                      # the photon arrives moving along -d
    p_t, p_s = lower(T, k_t[0] + 0 * a[0], k_s)
    N = X.size
    return (np.repeat(x[:, None], N, 1), np.full(N, cam['t']), p_s, p_t, np.ones(N))


# --- Workers ------------------------------------------------------------------------
_W = {}


def _init(p, binary):
    _W['p'], _W['binary'] = p, binary


def _trace(x, t, p_s, p_t, E, r_escape):
    o = mg.trace_from(_W['p'], _W['binary'], x, t, p_s, p_t, E, r_escape)
    return o['rho'], o['phi'], o['t'], o['n_cross'], o['opz'], o['captor']


def trace_frame(pool, rays, r_escape, chunks):
    x, t, p_s, p_t, E = rays
    rng = np.random.default_rng(0)
    parts = np.array_split(rng.permutation(len(t)), chunks)  # mixed, for even loads
    futs = [pool.submit(_trace, x[:, c], t[c], p_s[:, c], p_t[c], E[c], r_escape) for c in parts]
    return parts, futs


def collect(parts, futs, n):
    out = [np.full(n, np.nan) for _ in range(6)]
    for c, f in zip(parts, futs):
        for o, v in zip(out, f.result()):
            o[c] = v
    return out


def frame_fields(p, traced, shape):
    """Per screen point: which image layer of the disk it shows (-1 = none), the disk
    radius and azimuth there, the time the light left the disk, and the observed brightness."""
    rho, phi, t_em, ncr, opz, cap = traced
    hit = ~np.isnan(rho)
    layer = np.where(hit, np.minimum(np.nan_to_num(ncr), LAYERS[-1]), -1).astype(int)
    with np.errstate(invalid='ignore', divide='ignore'):
        F = np.where(hit, mg.disk_flux(p, np.where(hit, rho, 10.0)) / opz**4, 0.0)
    F = np.nan_to_num(F)
    return tuple(v.reshape(shape) for v in (layer, rho, phi, t_em, F, np.nan_to_num(cap, nan=-1)))


def add_rim(dens, layer, cap, pixel):
    """Raise the ink density on disk pixels next to a shadow (pixels whose rays fell into
    a hole), so each shadow gets a bright outline as in the stills."""
    shadow = cap >= 0
    if not shadow.any():
        return dens
    d = ndimage.distance_transform_edt(~shadow) * pixel
    return np.where(layer >= 0, np.maximum(dens, RIM * np.exp(-d / RIM_WIDTH)), dens)


# --- Main ---------------------------------------------------------------------------------
def main():
    from luminet_animation import ffmpeg_command, lower_priority
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='binary_flyby.mp4')
    ap.add_argument('--periapsis', type=float, default=15.0, help='closest approach to the center of mass, in M')
    ap.add_argument('--orbit-tilt', type=float, default=15.0, help="angle between the camera's orbit and the disk, in degrees")
    ap.add_argument('--retrograde-orbit', action='store_true', help='fly against the disk rotation')
    ap.add_argument('--start-radius', type=float, default=60.0, help='where the flight starts and ends, in M')
    ap.add_argument('--fov', type=float, default=40.0, help='horizontal field of view, in degrees')
    ap.add_argument('--duration', type=float, default=20.0, help='video length, in seconds')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--pacing', type=float, default=1.0,
                    help='slow-motion near the binary: 0 = the camera clock runs evenly, 1 = in proportion to distance')
    ap.add_argument('--disk-rotation', type=float, default=1.0, help='speed of the dots\' rotation with the disk (1 = physical)')
    ap.add_argument('--trace-width', type=int, default=400, help='ray-traced grid width (height = width / aspect)')
    ap.add_argument('--frames', type=float, nargs='*', default=None,
                    help='instead of a video, save single frames as PNGs at these points of the flight (0 to 1)')
    ap.add_argument('--warm', type=int, default=2 * fb.FADE,
                    help='with --frames, frames run before each saved one so the dots settle (0 = draw each frame fresh, like a still)')
    ap.add_argument('--range', type=float, nargs=2, default=None, metavar=('START', 'END'),
                    help='only render this part of the flight, as fractions from 0 to 1')
    ap.add_argument('--texture', type=float, default=0.5, help='paper texture strength (1 = as in the stills)')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--crf', type=float, default=12)
    ap.add_argument('--x264-preset', default='slow')
    ap.add_argument('--pix-fmt', default='yuv420p')
    ap.add_argument('--lossless', action='store_true')
    bh.add_param_args(ap, mg.PRESETS)
    args = ap.parse_args()
    lower_priority()
    p = mg.params_from_args(args).resolved()
    reach = max(p.separation * p.mass_ratio, p.separation) / (1 + p.mass_ratio) + 2 / (1 + p.mass_ratio)
    dataclasses.replace(p, r_in=max(p.r_in, reach + 1e-6, 6.0)).validate()   # r_in is checked below
    if p.r_in <= reach:
        print(f"note: the disk's inner edge ({p.r_in}M) is inside the holes' orbits")
    p = dataclasses.replace(p, view=VIEW)

    # the holes are at p.phase when the camera passes closest (t = 0)
    t_far = parabola_start(args.periapsis, args.orbit_tilt, args.start_radius)[2]
    binary = mg.make_binary(p, t_ref=0.0, t_range=(3 * t_far - 300, -2 * t_far + 100))
    t0 = time.time()
    path = camera_path(binary, args.periapsis, args.orbit_tilt, args.start_radius, not args.retrograde_orbit)
    tau = path['tau']
    print(f"camera path: closest approach {path['r_min']:.2f}M at t = {path['t_peri']:.1f}M, "
          f"{tau[-1]:.0f}M of proper time, t from {path['t'][0]:.0f} to {path['t'][-1]:.0f}M ({time.time() - t0:.0f} s)")
    n_frames = round(args.duration * args.fps)
    r = path['r']
    s = np.concatenate([[0], np.cumsum(np.diff(tau) / (0.5 * (r[1:] + r[:-1]) / args.periapsis)**args.pacing)])
    taus = np.interp(np.linspace(0, s[-1], n_frames), s, tau)

    Wt = args.trace_width; Ht = round(Wt / p.aspect)
    xs = np.linspace(-VIEW, VIEW, Wt); ys = np.linspace(-VIEW / p.aspect, VIEW / p.aspect, Ht)
    X, Y = np.meshgrid(xs, ys)
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    pool = ProcessPoolExecutor(workers, initializer=_init, initargs=(p, binary))

    def rays_for(tau_f, X, Y):
        cam = camera_at(binary, path, tau_f)
        return cam, camera_rays(cam, X, Y, args.fov), max(1.3 * cam['r'], 40.0)

    print("exposure pass...", flush=True)
    xl = np.linspace(-VIEW, VIEW, 160); yl = np.linspace(-VIEW / p.aspect, VIEW / p.aspect, 80)
    XL, YL = np.meshgrid(xl, yl)
    sample_frames = np.unique(np.linspace(0, n_frames - 1, min(n_frames, 40)).round().astype(int))
    refs = []
    for fi in sample_frames:
        cam, rays, resc = rays_for(taus[fi], XL, YL)
        layer, R, PHI, TEM, F, CAP = frame_fields(p, collect(*trace_frame(pool, rays, resc, workers * 2), XL.size), XL.shape)
        refs.append(fb.exposure_reference(p, F, layer))
    log_ref = np.log(np.array(refs))
    log_ref = np.interp(np.arange(n_frames), sample_frames, np.where(np.isnan(log_ref), np.nanmedian(log_ref), log_ref))
    ref = np.exp(ndimage.gaussian_filter1d(log_ref, 0.5 * args.fps, mode='nearest'))
    print(f"  done ({time.time() - t0:.0f} s)", flush=True)

    RES = 1
    W, H = round(bh.FIG_W * 200 * RES), round(bh.FIG_W / p.aspect * 200 * RES)
    tex = bh.Texturer(H, W, RES, p, strength=args.texture)
    blur = p.ink_blur * RES
    dots = fb.Dots(p, p.seed, args.disk_rotation, layers=LAYERS)
    if args.frames is not None and args.warm == 0:
        fb.FADE = 1                                      # no settling frames: draw dots at full size at once

    if args.frames is not None:
        frame_list = [int(round(fr * (n_frames - 1))) for fr in args.frames]
        warm = args.warm
    else:
        first, last = 0, n_frames
        if args.range:
            first, last = round(args.range[0] * (n_frames - 1)), round(args.range[1] * (n_frames - 1)) + 1
        frame_list = list(range(first, last))
        warm = 0
    enc = None
    if args.frames is None:
        enc = subprocess.Popen(ffmpeg_command(args.out, W, H, args.fps, args), stdin=subprocess.PIPE)
    schedule = []
    for fi in frame_list:
        schedule += [(g, g == fi) for g in range(max(0, fi - warm), fi + 1)] if warm else [(fi, True)]
    pending = None
    t1 = time.time()
    for k, (fi, output) in enumerate(schedule):
        if pending is None:
            cam, rays, resc = rays_for(taus[fi], X, Y)
            pending = (cam, trace_frame(pool, rays, resc, workers * 4))
        cam, job = pending
        traced = collect(*job, X.size)
        if k + 1 < len(schedule):                       # start tracing the next frame while this one is drawn
            ncam, nrays, nresc = rays_for(taus[schedule[k + 1][0]], X, Y)
            pending = (ncam, trace_frame(pool, nrays, nresc, workers * 4))
        else:
            pending = None
        layer, R, PHI, TEM, F, CAP = frame_fields(p, traced, X.shape)
        dens = add_rim(fb.density_field(p, R, F, layer, ref[fi]), layer, CAP, xs[1] - xs[0])
        if args.frames is not None and args.warm == 0:
            dots = fb.Dots(p, p.seed, args.disk_rotation, layers=LAYERS)   # each preview drawn fresh
        pos, size, ang, off = dots.step(fi, (xs, ys), layer, R, PHI, TEM, dens, cam['t'])
        if not output:
            continue
        m = fb.ink_coverage(pos, size, ang, off, p, W, H, 200 * RES)
        img = tex(ndimage.gaussian_filter(m, blur))
        if enc:
            enc.stdin.write(img.tobytes())
        else:
            from PIL import Image
            name = os.path.splitext(args.out)[0] + f"_{fi / (n_frames - 1):.3f}.png"
            Image.fromarray(img).save(name)
            print(f"  saved {name}")
        done = k + 1
        el = time.time() - t1
        print(f"  frame {fi}/{n_frames}: camera at r = {cam['r']:.1f}M, t = {cam['t']:.0f}M, "
              f"{dots.stats['alive']} dots, {el:.0f} s elapsed, ~{el / done * (len(schedule) - done) / 60:.0f} min left",
              flush=True)
    if enc:
        enc.stdin.close()
        if enc.wait() != 0:
            sys.exit("ffmpeg failed")
    pool.shutdown()
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
