"""
Animate the Luminet recreation being inked, one dot at a time.

Dots are drawn from the inside out along a short pen path: the drawing is split
into thin contour bands around the hole (circles near the hole, stretching to
the frame's 2:1 shape farther out), each band is swept around in alternating
directions, and the whole route is then shortened with 2-opt, a standard
traveling-salesman improvement step.

Output is 60 fps at half the still's DPI (2800 x 1400 px). Each frame is built
exactly like the still: ink is stamped into a coverage map, softened with the
same blur and given the same paper texture. The texture is fixed from frame to
frame, so the only change between frames is the new dot. That keeps the
encoder's job easy: it spends its bits on the dots instead of on noise.

    python luminet_animation.py                   # full video, one dot per frame
    python luminet_animation.py --dots-per-frame 4
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
SUPERSAMPLE = 8          # subsamples per pixel side for antialiased dot edges


def disk_coverage(cy, cx, radius, r0, c0, K):
    """Fraction of each pixel in the K x K patch at (r0, c0) covered by each disk."""
    s = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE
    sub = (np.arange(K)[:, None] + s[None, :]).ravel()          # subsample offsets along one axis
    yy = r0[:, None] + sub[None, :] - cy[:, None]               # (n, K*S)
    xx = c0[:, None] + sub[None, :] - cx[:, None]
    inside = yy[:, :, None]**2 + xx[:, None, :]**2 <= radius[:, None, None]**2
    return inside.reshape(len(cy), K, SUPERSAMPLE, K, SUPERSAMPLE).mean(axis=(2, 4)).astype(np.float32)


def make_stamps(L, H, W, dpi):
    """Per-dot ink coverage patches, matching matplotlib's scatter of dots and lobes."""
    px_pt = dpi / 72
    def to_px(xy):
        return (L.VIEW / 2 - xy[:, 1]) / L.VIEW * H, (xy[:, 0] + L.VIEW) / (2 * L.VIEW) * W
    cy, cx = to_px(L.dots); ly, lx = to_px(L.lobe)
    r_dot = 0.5 * L.size * px_pt; r_lobe = 0.5 * L.LOBE_SCALE * L.size * px_pt
    K = int(np.ceil(2 * (r_dot.max() + np.hypot(cy - ly, cx - lx).max()))) + 3
    r0 = np.floor(cy - K / 2).astype(int) + 1; c0 = np.floor(cx - K / 2).astype(int) + 1
    stamps = np.empty((len(cy), K, K), np.float32)
    for s in range(0, len(cy), 2000):
        sl = slice(s, s + 2000)
        a_dot = disk_coverage(cy[sl], cx[sl], r_dot[sl], r0[sl], c0[sl], K)
        a_lobe = disk_coverage(ly[sl], lx[sl], r_lobe[sl], r0[sl], c0[sl], K)
        stamps[sl] = 1 - (1 - a_dot) * (1 - a_lobe)          # two overlapping opaque marks
    return r0, c0, stamps


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None, help='output file (default luminet_animation.mp4, or .mkv with --lossless)')
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--dots-per-frame', type=float, default=1.0,
                    help='dots added per frame (1 = one dot per frame, about 6.8 minutes)')
    ap.add_argument('--hold-start', type=float, default=0.5, help='seconds of blank paper first')
    ap.add_argument('--hold-end', type=float, default=4.0, help='seconds to hold the finished drawing')
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
    r0, c0, stamps = make_stamps(L, H, W, dpi)
    r0, c0, stamps = r0[order], c0[order], stamps[order]
    K = stamps.shape[1]

    tex = L.Texturer(H, W, RES)
    blur_sigma = L.INK_BLUR * RES
    reach = int(np.ceil(4 * blur_sigma))                  # gaussian_filter's kernel radius
    P = K + 1                                             # padding so stamps never fall off the array
    M = np.zeros((H + 2 * P, W + 2 * P), np.float32)      # raw ink coverage, before blur
    m = M[P:-P, P:-P]
    frame = tex(ndimage.gaussian_filter(m, blur_sigma))  # blank textured paper

    def add_dot(i):
        y0, x0 = r0[i] + P, c0[i] + P
        M[y0:y0 + K, x0:x0 + K] = 1 - (1 - M[y0:y0 + K, x0:x0 + K]) * (1 - stamps[i])
        # re-blur and re-texture only the pixels this dot can reach; identical to
        # doing the whole frame, because both steps are local
        ya, yb = max(r0[i] - reach, 0), min(r0[i] + K + reach, H)
        xa, xb = max(c0[i] - reach, 0), min(c0[i] + K + reach, W)
        if ya >= yb or xa >= xb:
            return
        wa, wb = max(ya - reach, 0), min(yb + reach, H)
        va, vb = max(xa - reach, 0), min(xb + reach, W)
        blurred = ndimage.gaussian_filter(m[wa:wb, va:vb], blur_sigma)[ya - wa:yb - wa, xa - va:xb - va]
        sub = (slice(ya, yb), slice(xa, xb))
        frame[sub] = tex.patch(blurred, sub)

    n = len(order)
    n_draw = int(np.ceil(n / args.dots_per_frame))
    n_frames = round(args.hold_start * args.fps) + n_draw + round(args.hold_end * args.fps)
    if args.seconds is not None:
        n_frames = min(n_frames, round(args.seconds * args.fps))
    print(f"{n_frames} frames ({n_frames / args.fps:.1f} s) -> {out}")
    enc = subprocess.Popen(ffmpeg_command(out, W, H, args.fps, args), stdin=subprocess.PIPE)
    drawn = 0; t0 = time.time()
    for f in range(n_frames):
        k = f - round(args.hold_start * args.fps) + 1    # drawing frames are numbered from 1
        target = min(n, int(np.floor(k * args.dots_per_frame + 1e-9))) if k > 0 else 0
        while drawn < target:
            add_dot(drawn); drawn += 1
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"\nDone in {time.time() - t0:.0f} s: {drawn} of {n} dots drawn")
    if args.last_frame:
        from PIL import Image
        Image.fromarray(frame).save(args.last_frame)


if __name__ == '__main__':
    main()
