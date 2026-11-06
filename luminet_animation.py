"""
Animate the Luminet recreation being inked, a few dots at a time.

Dots are drawn from the inside out along a short pen path: the drawing is split
into thin contour bands around the hole (circles near the hole, stretching to
the frame's 2:1 shape farther out), each band is swept around in alternating
directions, and the whole route is then shortened with 2-opt, a standard
traveling-salesman improvement step.

Each drop lands wet: a little smaller than its final size and domed, so it is
shaded like a droplet with a small glint. Over a few seconds it flattens and
spreads to its final size, ending exactly as the dry dot in the still.

Output is 60 fps at half the still's DPI (2800 x 1400 px). Frames are built like
the still: ink is stamped into a coverage map, softened with the same blur and
given the paper texture (at half strength by default). The texture is fixed from
frame to frame, so only the drops change. That keeps the encoder's job easy: it
spends its bits on the dots instead of on noise.

    python luminet_animation.py                   # full video, 3 dots per frame
    python luminet_animation.py --dots-per-frame 1
    python luminet_animation.py --seconds 10      # quick preview of the start
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def lower_priority():
    """Run at below-normal CPU priority; ffmpeg inherits it."""
    if sys.platform == 'win32':
        import ctypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x4000
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
    else:
        os.nice(5)


# --- Drawing order ---------------------------------------------------------
ROUND_UNTIL = 6.0        # contours are circles out to this radius (in M)...
FRAME_ASPECT = 2.0       # ...then stretch to the frame's width:height ratio
BAND_WIDTH = 2.0         # contour band width, in local dot spacings
TWO_OPT_WINDOW = 60      # how far ahead (in dots) 2-opt looks for a better link


def contour_radius(x, y, half_height):
    """Label each point with the contour it sits on. A contour of radius rho is an
    ellipse with half-height rho and half-width a(rho): a circle up to ROUND_UNTIL,
    widening to FRAME_ASPECT * rho by the time it reaches the frame's top edge."""
    def a(rho):
        t = np.clip((rho - ROUND_UNTIL) / (half_height - ROUND_UNTIL), 0, 1)
        return rho * (1 + t * (FRAME_ASPECT - 1))
    r = np.hypot(x, y)
    lo, hi = r / FRAME_ASPECT, r.copy()          # the contour radius always lies in between
    for _ in range(50):                          # bisection: x^2/a^2 + y^2/rho^2 = 1
        mid = 0.5 * (lo + hi)
        with np.errstate(divide='ignore', invalid='ignore'):
            outside = (x / a(mid))**2 + (y / mid)**2 > 1
        lo = np.where(outside, mid, lo); hi = np.where(outside, hi, mid)
    return 0.5 * (lo + hi)


def path_length(P):
    return np.linalg.norm(np.diff(P, axis=0), axis=1).sum()


def two_opt(P, order, window, max_passes=20):
    """Windowed 2-opt on an open path: reverse a stretch whenever that shortens it."""
    order = order.copy(); Q = P[order]
    n = len(Q)
    for p in range(max_passes):
        improved = 0
        for i in range(n - 3):
            j = np.arange(i + 2, min(i + window, n - 1))
            a, b = Q[i], Q[i + 1]
            c, d = Q[j], Q[j + 1]
            gain = (np.linalg.norm(a - b) + np.linalg.norm(c - d, axis=1)
                    - np.linalg.norm(c - a, axis=1) - np.linalg.norm(d - b, axis=1))
            k = gain.argmax()
            if gain[k] > 1e-12:
                jj = j[k]
                Q[i + 1:jj + 1] = Q[i + 1:jj + 1][::-1].copy()
                order[i + 1:jj + 1] = order[i + 1:jj + 1][::-1].copy()
                improved += 1
        print(f"  2-opt pass {p + 1}: {improved} improvements, path {path_length(Q):.1f} M")
        if not improved:
            break
    return order


def drawing_order(P, half_height):
    x, y = P.T
    rho = contour_radius(x, y, half_height)
    theta = np.arctan2(y, x)
    # local dot spacing as a function of contour radius, so bands are a fixed
    # number of dot spacings wide in both the dense and the sparse parts
    d_nn = cKDTree(P).query(P, k=2)[0][:, 1]
    edges = np.quantile(rho, np.linspace(0, 1, 121))
    which = np.clip(np.searchsorted(edges, rho, side='right') - 1, 0, 119)
    spacing = np.array([np.median(d_nn[which == i]) for i in range(120)])
    centers = 0.5 * (edges[1:] + edges[:-1])
    grid = np.linspace(0, rho.max() + 1e-6, 20000)
    steps = np.concatenate([[0], np.cumsum(np.diff(grid) / np.interp(grid[1:], centers, spacing))])
    band = np.floor(np.interp(rho, grid, steps) / BAND_WIDTH).astype(int)
    # sweep each band around the hole, alternating direction, so each band
    # starts where the previous one ended (at theta = +-pi, the left side)
    key = np.where(band % 2 == 0, theta, -theta)
    order = np.lexsort((key, band))
    print(f"  {band.max() + 1} bands, sweep path {path_length(P[order]):.1f} M")
    order = two_opt(P, order, TWO_OPT_WINDOW)
    hops = np.linalg.norm(np.diff(P[order], axis=0), axis=1)
    print(f"  median hop {np.median(hops):.3f} M vs median nearest-neighbor distance "
          f"{np.median(d_nn):.3f} M; longest hop {hops.max():.2f} M")
    return order


# --- Ink stamps --------------------------------------------------------------
SUPERSAMPLE = 8          # subsamples per pixel side for antialiased dry dot edges
WET_SUPERSAMPLE = 6      # ...and for the wet drops (cheaper; they are always changing)
WET_LEVELS = 12          # wetness levels precomputed per drop; frames interpolate between them

# How a wet drop looks. Wetness w runs from 1 (just landed) to 0 (dry).
WET_SHRINK = 0.15        # a fresh drop is this much smaller than its final size
DOME = 0.6               # height / radius of a fresh drop; it flattens as it dries
LIGHT = np.array([-0.55, -0.65, 1.0]) / np.linalg.norm([-0.55, -0.65, 1.0])   # from upper left (x right, y down, z to viewer)
HALF = (LIGHT + [0, 0, 1]) / np.linalg.norm(LIGHT + [0, 0, 1])
DIFFUSE = 0.55           # how strongly the dome's slope lightens / darkens the ink
GLINT, GLINT_POWER = 0.45, 14   # specular highlight strength and tightness


def drop_geometry(L, H, W, dpi):
    """Pixel positions and radii of every dot and its lobe, and the stamp grid."""
    px_pt = dpi / 72
    def to_px(xy):
        return (L.VIEW / 2 - xy[:, 1]) / L.VIEW * H, (xy[:, 0] + L.VIEW) / (2 * L.VIEW) * W
    cy, cx = to_px(L.dots); ly, lx = to_px(L.lobe)
    r_dot = 0.5 * L.size * px_pt; r_lobe = 0.5 * L.LOBE_SCALE * L.size * px_pt
    K = int(np.ceil(2 * (r_dot.max() + np.hypot(cy - ly, cx - lx).max()))) + 3
    r0 = np.floor(cy - K / 2).astype(int) + 1; c0 = np.floor(cx - K / 2).astype(int) + 1
    return dict(cy=cy, cx=cx, ly=ly, lx=lx, r_dot=r_dot, r_lobe=r_lobe, r0=r0, c0=c0, K=K)


def drop_stamps(g, sl, w, S):
    """Ink coverage, shading gain and glint for dots sl at wetness w, each a K x K
    patch. A drop is the dot and its lobe, each a dome; the higher dome wins where
    they overlap. At w = 0 the drop is flat and full size: exactly the dry dot."""
    cy, cx, r0, c0, K = g['cy'][sl], g['cx'][sl], g['r0'][sl], g['c0'][sl], g['K']
    s = 1 - WET_SHRINK * w
    ly = cy + s * (g['ly'][sl] - cy); lx = cx + s * (g['lx'][sl] - cx)   # lobe moves in as the drop shrinks
    R1 = (s * g['r_dot'][sl]).astype(np.float32); R2 = (s * g['r_lobe'][sl]).astype(np.float32)
    sub = (np.arange(K)[:, None] + (np.arange(S) + 0.5) / S).ravel()
    n = len(cy)
    def dome(yc, xc, R):
        dy = (r0[:, None] + sub[None, :] - yc[:, None]).astype(np.float32)[:, :, None]
        dx = (c0[:, None] + sub[None, :] - xc[:, None]).astype(np.float32)[:, None, :]
        q = (dy**2 + dx**2) / (R**2)[:, None, None]
        return dx, dy, q, q <= 1
    pool = lambda v: v.reshape(n, K, S, K, S).mean(axis=(2, 4), dtype=np.float32)
    dx1, dy1, q1, in1 = dome(cy, cx, R1)
    dx2, dy2, q2, in2 = dome(ly, lx, R2)
    a = 1 - (1 - pool(in1)) * (1 - pool(in2))         # two overlapping opaque marks, as in the still
    if w == 0:
        z = np.zeros_like(a)
        return a, z, z
    h1 = np.where(in1, np.sqrt(np.maximum(1 - q1, 0)) * R1[:, None, None], -1)
    h2 = np.where(in2, np.sqrt(np.maximum(1 - q2, 0)) * R2[:, None, None], -1)
    use1 = h1 >= h2
    def slope(dx, dy, q, R):          # gradient of the dome height DOME*w * R * sqrt(1 - q)
        k = -DOME * w / (R[:, None, None] * np.sqrt(np.maximum(1 - q, 0.04)))
        return k * dx, k * dy
    hx1, hy1 = slope(dx1, dy1, q1, R1); hx2, hy2 = slope(dx2, dy2, q2, R2)
    hx = np.where(use1, hx1, hx2); hy = np.where(use1, hy1, hy2)
    nz = 1 / np.sqrt(1 + hx**2 + hy**2)
    nx, ny = -hx * nz, -hy * nz
    lit = np.maximum(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0)
    gain = DIFFUSE * (lit / LIGHT[2] - 1)              # 0 on a flat surface
    spec = np.maximum(nx * HALF[0] + ny * HALF[1] + nz * HALF[2], 0)
    glint = GLINT * w * np.maximum(spec**GLINT_POWER - HALF[2]**GLINT_POWER, 0)
    inside = in1 | in2
    return a, pool(inside * gain), pool(inside * glint)


class Stamps:
    """Dry stamps (float32), plus wet stamps at WET_LEVELS wetness levels (float16).
    Made in chunks just before the drops land, so previews only pay for what they show."""
    CHUNK = 300
    def __init__(self, g):
        self.g = g; N, K = len(g['cy']), g['K']
        self.dry = np.empty((N, K, K), np.float32)
        self.A = np.empty((N, WET_LEVELS, K, K), np.float16)
        self.G = np.zeros_like(self.A); self.SP = np.zeros_like(self.A)
        self.ready = 0
    def ensure(self, upto):
        while self.ready < min(upto, len(self.dry)):
            sl = slice(self.ready, self.ready + self.CHUNK)
            self.dry[sl] = drop_stamps(self.g, sl, 0.0, SUPERSAMPLE)[0]
            self.A[sl, 0] = self.dry[sl]
            for j, w in enumerate(np.linspace(0, 1, WET_LEVELS)[1:], 1):
                self.A[sl, j], self.G[sl, j], self.SP[sl, j] = drop_stamps(self.g, sl, w, WET_SUPERSAMPLE)
            self.ready += self.CHUNK


# --- Encoding ------------------------------------------------------------------
def ffmpeg_command(out, W, H, fps, args):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-stats',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-']
    if args.lossless:
        # exact RGB, for a master to re-encode from; plays in VLC/ffplay, not browsers
        cmd += ['-c:v', 'libx264rgb', '-qp', '0', '-preset', 'medium', '-pix_fmt', 'rgb24']
    else:
        # The drawing is all tiny dots on a grainy background, the worst case for a
        # video encoder, so:
        #   - high quality (CRF 12) and -tune grain, which stops x264 from smoothing
        #     the paper texture and the edges of the dots away;
        #   - an explicit BT.709 RGB->YUV matrix, tagged in the file, so players
        #     don't shift the colors;
        #   - 4:2:0 chroma, which is safe here because the image is nearly gray;
        #     pass --pix-fmt yuv444p for full color resolution (less compatible).
        cmd += ['-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd',
                '-c:v', 'libx264', '-preset', args.preset, '-crf', str(args.crf), '-tune', 'grain',
                '-pix_fmt', args.pix_fmt, '-g', str(10 * fps),
                '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                '-color_range', 'tv', '-movflags', '+faststart']
    return cmd + [out]


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help='output file (default luminet_animation.mp4, or .mkv with --lossless)')
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--dots-per-frame', type=float, default=3.0, help='dots added per frame')
    ap.add_argument('--dry-seconds', type=float, default=5.0, help='how long each drop takes to dry')
    ap.add_argument('--texture', type=float, default=0.5,
                    help='strength of the paper grain and tone, relative to the still (1 = same)')
    ap.add_argument('--hold-start', type=float, default=0.5, help='seconds of blank paper first')
    ap.add_argument('--hold-end', type=float, default=4.0, help='seconds to hold the finished, dry drawing')
    ap.add_argument('--seconds', type=float, default=None, help='stop after this many seconds (for previews)')
    ap.add_argument('--crf', type=float, default=12, help='x264 quality: lower is better and bigger')
    ap.add_argument('--preset', default='slow', help='x264 speed/efficiency preset')
    ap.add_argument('--pix-fmt', default='yuv420p', help='yuv420p (plays everywhere) or yuv444p')
    ap.add_argument('--lossless', action='store_true', help='lossless RGB H.264 master in .mkv')
    ap.add_argument('--last-frame', default=None, help='also save the final frame as this PNG')
    args = ap.parse_args()
    out = args.out or ('luminet_animation.mkv' if args.lossless else 'luminet_animation.mp4')
    lower_priority()

    print("Placing dots (same as the still)...")
    import luminet_1979 as L
    RES = L.RES_SCALE / 2                                 # half the still's DPI
    dpi = 200 * RES
    W, H = round(L.FIG_W * dpi), round(L.FIG_H * dpi)
    print(f"Rendering {len(L.dots)} dots at {W}x{H}, {args.fps} fps")

    print("Ordering dots...")
    order = drawing_order(L.dots, L.VIEW / 2)
    print("Making ink stamps...")
    geom = drop_geometry(L, H, W, dpi)
    geom = {k: (v[order] if isinstance(v, np.ndarray) else v) for k, v in geom.items()}
    stamps = Stamps(geom)
    dry, A, G, SP = stamps.dry, stamps.A, stamps.G, stamps.SP
    r0, c0, K = geom['r0'], geom['c0'], geom['K']

    tex = L.Texturer(H, W, RES, strength=args.texture)
    blur_sigma = L.INK_BLUR * RES
    reach = int(np.ceil(4 * blur_sigma))                  # gaussian_filter's kernel radius
    P = K + 1                                             # padding so stamps never fall off the arrays
    Hp, Wp = H + 2 * P, W + 2 * P
    M = np.zeros((Hp, Wp), np.float32)                    # ink coverage of the dry dots, before blur
    m = M[P:-P, P:-P]
    base = tex(ndimage.gaussian_filter(m, blur_sigma))    # the frame with only the dry dots
    # Wet drops are redrawn every frame: T is how much paper they leave showing,
    # Gm and Sm their shading gain and glint.
    T = np.ones((Hp, Wp), np.float32); Gm = np.zeros_like(T); Sm = np.zeros_like(T)
    Tf, Gf, Sf = T.reshape(-1), Gm.reshape(-1), Sm.reshape(-1)
    Ti, Gi, Si = T[P:-P, P:-P], Gm[P:-P, P:-P], Sm[P:-P, P:-P]
    offsets = (np.arange(K)[:, None] * Wp + np.arange(K)[None, :]).ravel()
    flat0 = (r0 + P) * Wp + (c0 + P)                      # flat index of each stamp's corner

    def render(region, wet=False):
        """Blur + texture (+ wet drops and their shading) for one region of the frame.
        Both steps are local, so this matches rendering the whole frame."""
        ya, yb, xa, xb = region
        wa, wb, va, vb = max(ya - reach, 0), min(yb + reach, H), max(xa - reach, 0), min(xb + reach, W)
        cov = m[wa:wb, va:vb]
        if wet:
            cov = 1 - (1 - cov) * Ti[wa:wb, va:vb]
        blurred = ndimage.gaussian_filter(cov, blur_sigma)[ya - wa:yb - wa, xa - va:xb - va]
        rgb = tex.patch(blurred, (slice(ya, yb), slice(xa, xb)))
        if wet:
            gain, glint = Gi[ya:yb, xa:xb], Si[ya:yb, xa:xb]
            rgb = np.clip(rgb * (1 + gain[..., None]) + 255 * glint[..., None], 0, 255).astype(np.uint8)
        return rgb

    def dry_dot(i):
        y0, x0 = r0[i] + P, c0[i] + P
        M[y0:y0 + K, x0:x0 + K] = 1 - (1 - M[y0:y0 + K, x0:x0 + K]) * (1 - dry[i])
        region = (max(r0[i] - reach, 0), min(r0[i] + K + reach, H), max(c0[i] - reach, 0), min(c0[i] + K + reach, W))
        if region[0] < region[1] and region[2] < region[3]:
            base[region[0]:region[1], region[2]:region[3]] = render(region)

    TILE = 32                                             # must be >= K + 2 * reach
    n_ty, n_tx = -(-H // TILE), -(-W // TILE)

    n = len(order)
    start = round(args.hold_start * args.fps)
    appear = start + np.floor(np.arange(n) / args.dots_per_frame + 1e-9).astype(int)   # frame each drop lands
    dry_frames = max(1, round(args.dry_seconds * args.fps))
    n_frames = appear[-1] + dry_frames + round(args.hold_end * args.fps)
    if args.seconds is not None:
        n_frames = min(n_frames, round(args.seconds * args.fps))
    print(f"{n_frames} frames ({n_frames / args.fps:.1f} s) -> {out}")
    enc = subprocess.Popen(ffmpeg_command(out, W, H, args.fps, args), stdin=subprocess.PIPE)
    dried = 0; prev_idx = None; frame = base; t0 = time.time()
    for f in range(n_frames):
        while dried < n and appear[dried] + dry_frames <= f:
            dry_dot(dried); dried += 1
        landed = np.searchsorted(appear, f, side='right')      # drops on the paper so far
        stamps.ensure(landed)
        if prev_idx is not None:                               # clear last frame's wet drops
            Tf[prev_idx] = 1; Gf[prev_idx] = 0; Sf[prev_idx] = 0; prev_idx = None
        wet = np.arange(dried, landed)
        if len(wet) == 0:
            frame = base
        else:
            wetness = 1 - smoothstep((f - appear[wet]) / dry_frames)
            lv = wetness * (WET_LEVELS - 1)
            lo = np.minimum(np.floor(lv).astype(int), WET_LEVELS - 2)
            t = (lv - lo)[:, None, None].astype(np.float32)
            a = A[wet, lo] * (1 - t) + A[wet, lo + 1] * t
            gn = G[wet, lo] * (1 - t) + G[wet, lo + 1] * t
            gl = SP[wet, lo] * (1 - t) + SP[wet, lo + 1] * t
            idx = (flat0[wet][:, None] + offsets[None, :]).ravel()
            np.multiply.at(Tf, idx, (1 - a).ravel())
            np.add.at(Gf, idx, gn.ravel()); np.add.at(Sf, idx, gl.ravel())
            prev_idx = idx
            # rebuild only the tiles the wet drops (plus blur reach) touch; a tile is
            # at least as big as a drop, so checking each drop's four corners finds them all
            ya, yb = r0[wet] - reach, r0[wet] + K + reach
            xa, xb = c0[wet] - reach, c0[wet] + K + reach
            ty = np.clip(np.concatenate([ya, ya, yb, yb]) // TILE, 0, n_ty - 1)
            tx = np.clip(np.concatenate([xa, xb, xa, xb]) // TILE, 0, n_tx - 1)
            frame = base.copy()
            for code in np.unique(ty * n_tx + tx):
                y, x = divmod(int(code), n_tx)
                region = (y * TILE, min((y + 1) * TILE, H), x * TILE, min((x + 1) * TILE, W))
                frame[region[0]:region[1], region[2]:region[3]] = render(region, wet=True)
        enc.stdin.write(frame.tobytes())
        if f % 600 == 0:
            print(f"  frame {f}/{n_frames}, {len(wet)} wet drops, {time.time() - t0:.0f} s", flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"\nDone in {time.time() - t0:.0f} s: {min(landed, n)} of {n} dots landed, {dried} dry")
    if args.last_frame:
        from PIL import Image
        Image.fromarray(frame).save(args.last_frame)


if __name__ == '__main__':
    main()
