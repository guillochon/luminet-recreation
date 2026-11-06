"""
Recreating J.-P. Luminet's 1979 black hole image
("Image of a spherical black hole with thin accretion disk", A&A 75, 228).

Same physics as the original:
  1. Schwarzschild light bending (photon orbit eq. u'' = -u + 3Mu^2, M = 1)
  2. Thin Keplerian disk with Page-Thorne bolometric flux, inner edge at 6M
  3. Gravitational + Doppler shift: F_obs = F / (1+z)^4
  4. Primary image + secondary ("ghost") image that wraps over the hole
  5. Rendered the way Luminet drew it: dots, density ~ observed flux
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

M = 1.0
INCL = np.radians(80)          # observer 80 deg from disk axis (10 deg above plane), as in 1979
R_IN, R_OUT = 6.0, 400.0        # disk: innermost stable orbit -> outer edge

# --- 1. Integrate photon orbits u(phi) for a range of impact parameters b ---
b_vals = np.linspace(0.02, 60.0, 3500)
dphi = 0.002
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
interp = RegularGridInterpolator((phis, b_vals), U, bounds_error=False, fill_value=np.nan)

# --- 2. Disk physics ---
def page_thorne(r):
    # Eq. 15 prints sqrt(3)/3 on the log term; sqrt(3)/2 is the correct Page-Thorne
    # value and is what reproduces the paper's own numbers (peak at r=9.55M, 1.146e-4).
    x = r / M
    s3, s6 = np.sqrt(3), np.sqrt(6)
    return (3*M/(8*np.pi)) / ((x - 3) * x**2.5) * (
        np.sqrt(x) - s6 + (s3/2) * np.log((np.sqrt(x)+s3)*(s6-s3) / ((np.sqrt(x)-s3)*(s6+s3))))

# Luminet's drawing goes from dark shadow to dense dots abruptly, while Page-Thorne
# brightness falls to zero at r = 6M. SHARP_INNER_EDGE holds the intrinsic brightness
# at its peak value inside r = 9.55M (gravitational redshift still dims it inward).
SHARP_INNER_EDGE = True
R_PEAK = 9.55
EDGE_SOFTNESS = 0.4              # in M: how far into the disk the fade-in from the shadow lasts
def observed_flux(r, b, alpha):
    if SHARP_INNER_EDGE:
        r_src = np.maximum(r, R_PEAK)
        fade = 1 - np.exp(-np.maximum(r - R_IN, 0) / EDGE_SOFTNESS)
        one_plus_z = (1 + np.sqrt(M / r**3) * b * np.sin(INCL) * np.sin(alpha)) / np.sqrt(1 - 3*M/r)
        return fade * page_thorne(r_src) / one_plus_z**4
    one_plus_z = (1 + np.sqrt(M / r**3) * b * np.sin(INCL) * np.sin(alpha)) / np.sqrt(1 - 3*M/r)
    return page_thorne(r) / one_plus_z**4

# --- 3. Map every image-plane pixel onto the disk ---
VIEW = 36.0                    # half-width of the frame in M (height is half of this)
W, H = 1680, 840
xs = np.linspace(-VIEW, VIEW, W); ys = np.linspace(-VIEW/2, VIEW/2, H)
X, Y = np.meshgrid(xs, ys)
B = np.hypot(X, Y)
ALPHA = np.arctan2(X, -Y)                      # alpha = 0 points at the near side of the disk
cosA = np.cos(ALPHA)
gamma = np.arccos(cosA / np.sqrt(cosA**2 + 1/np.tan(INCL)**2))

flux = np.zeros_like(B)
RMAP = np.full_like(B, np.nan)   # disk radius seen at each pixel
# Secondary image. Luminet did not compute it fully: he drew it as a thin line at
# b = 5.25M and overestimated its brightness with r = 9.55M, b = 5.25M in Eqs. 15 & 19.
LUMINET_GHOST = False
GHOST_B, GHOST_WIDTH = 5.25, 0.12
for n in ((0,) if LUMINET_GHOST else (1, 0)):   # secondary first, primary overwrites (opaque disk)
    phi = gamma + n * np.pi
    un = interp(np.stack([phi.ravel(), B.ravel()], -1)).reshape(B.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = 1.0 / un
        hit = (r >= R_IN) & (r <= R_OUT)
        F = np.where(hit, observed_flux(np.where(hit, r, 10), B, ALPHA), 0)
    flux = np.where(hit, F, flux)
    RMAP = np.where(hit, r, RMAP)
    if n == 1: SEC = hit.copy()
    else: PRIM = hit
SEC = (SEC & ~PRIM) if not LUMINET_GHOST else np.zeros_like(PRIM)
if LUMINET_GHOST:
    SEC = (np.abs(B - GHOST_B) < GHOST_WIDTH) & np.isnan(RMAP)
    ghost = (np.abs(B - GHOST_B) < GHOST_WIDTH) & np.isnan(RMAP)
    flux = np.where(ghost, observed_flux(9.55, GHOST_B, ALPHA), flux)

# --- 4. Stipple it, like Luminet's India-ink dots ---
rng = np.random.default_rng(1979)
# Luminet plotted his dots along isoradials: curves of constant disk radius.
# Mimic that by only inking near a discrete set of radii, boosting those
# bands so the average dot density still follows the observed flux.
BANDED = True
RING_STEP = 1.44                 # each ring 44% farther out than the last
BAND_DEPTH = 0.10                # 0 = no banding, 1 = fully empty gaps between rings
weight = np.ones_like(B)
if BANDED:
    k = np.log(RMAP / R_IN) / np.log(RING_STEP)          # fractional ring index
    ring = 1 + BAND_DEPTH * np.cos(2 * np.pi * np.nan_to_num(k))   # peaks on each ring
    weight = np.where(np.isnan(RMAP), 1.0, ring)
DOT_GAMMA = 1.3                  # >1 thins the dim (right, outer) disk more than the bright left
DOT_GAIN = 2.2                   # >1 pushes mid-bright regions (above/below the shadow) toward saturation
DOT_CAP = 0.96                   # density ceiling: sets how packed the brightest, saturated region is
disk_density = np.minimum(DOT_GAIN * np.nan_to_num((flux / np.nanmax(flux)) ** DOT_GAMMA), DOT_CAP) * weight
disk_density = np.where(SEC, 0.0, disk_density)   # the ring band is inked separately, in rows
BACKGROUND_DENSITY = 0.0047     # faint speckle that dominates the darkest areas far from the hole
density = disk_density + np.where(np.isnan(RMAP), 0.0, BACKGROUND_DENSITY)   # on the disk only, none in the hole

# Hand-inked look: Luminet's dots are evenly spaced, not random, and a bit blobby.
# 1) Draw many candidate dots in proportion to brightness.
# 2) Poisson-disk elimination: walk through them in random order and keep a dot only
#    if no kept dot lies within its spacing radius. The radius shrinks where it's
#    brighter (spacing ~ 1/sqrt(density)), so brightness becomes dot spacing.
from scipy.spatial import cKDTree
SPACING_AT_PEAK = 0.05 * VIEW / 30         # dot spacing in the brightest region, in units of M
OVERSAMPLE = 8                   # candidates per final dot
density = np.minimum(density, 1.0); dens_max = 1.0   # boosted ring is capped, not used to normalize
# expected final dots ~ (brightness-weighted area) / spacing^2
weighted_area = (density / dens_max).mean() * (2*VIEW) * VIEW
n_cand = int(OVERSAMPLE * weighted_area / SPACING_AT_PEAK**2)

# Regular rows along the disk's inner edge, the way Luminet kept that edge clean:
# row 0 follows the edge with even spacing; each further row steps into the disk and
# gets more positional jitter, so by EDGE_ROWS rows in it blends into random placement.
EDGE_ROWS = 4
from scipy import ndimage
from skimage import measure
px_M = 2 * VIEW / (W - 1)
dist = ndimage.distance_transform_edt(PRIM) * px_M            # distance into the disk from its edge, in M
dens_inner = ndimage.maximum_filter(density, size=max(3, int(1.0 / px_M)))
s_rows = SPACING_AT_PEAK / np.sqrt(np.maximum(dens_inner, 1e-4))   # row-to-row gap (ignores the edge fade)
s_true = SPACING_AT_PEAK / np.sqrt(np.maximum(density, 1e-4))      # local dot spacing (includes the fade)
u = dist / s_rows
def at(field, x, y):
    j = np.clip(((x + VIEW) / (2*VIEW) * (W - 1)).round().astype(int), 0, W - 1)
    i = np.clip(((y + VIEW/2) / VIEW * (H - 1)).round().astype(int), 0, H - 1)
    return field[i, j]
edge_dots = []
for k in range(EDGE_ROWS):
    for c in measure.find_contours(u, 0.5 + 0.9 * k):
        cxs = np.interp(c[:, 1], np.arange(W), xs); cys = np.interp(c[:, 0], np.arange(H), ys)
        if np.hypot(cxs, cys).min() > 15: continue                 # only the edge around the hole
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
    cx = rng.uniform(-VIEW, VIEW, n); cy = rng.uniform(-VIEW/2, VIEW/2, n)
    ix = np.clip(((cx + VIEW) / (2*VIEW) * (W - 1)).round().astype(int), 0, W - 1)
    iy = np.clip(((cy + VIEW/2) / VIEW * (H - 1)).round().astype(int), 0, H - 1)
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
kept = np.array(kept)
dots = np.vstack([edge_dots, pts[kept]])
print("edge-row dots:", len(edge_dots))

# The secondary ring is inked separately, the way Luminet did it: a carefully dotted
# line along the ring's inner edge, plus a fan of evenly spaced (but not gridded) dots
# behind it where the ring broadens. The fan's width follows ring width x brightness,
# so it flares out on the bright lower left, barely appears on the lower right, and
# its dots are spaced together with the disk's so the fan merges into the disk.
RING_SPACING_LEFT = 0.17 * VIEW / 30        # distance between line dots at the left end, in M
RING_SPACING_RIGHT = 0.34 * VIEW / 30       # ...and at the right end: half as dense, as in his drawing
RING_SPACING = 0.5 * (RING_SPACING_LEFT + RING_SPACING_RIGHT)
RIGHT_FAN_DOTS = 4                          # second row of dots along the last few line dots on the right
FAN_UNIT = 0.19 * VIEW / 30                 # fan sizes are set in this unit, independent of the line spacing
FAN_MAX_WIDTH = 0.64 * 13 * 0.85 * FAN_UNIT    # fan width at its widest, where it meets the disk
FAN_TAPER = 0.6                             # <1 makes the fan widen more gradually
FAN_THRESHOLD = 0.12                        # below this (relative width x brightness) the ring stays 1 dot wide
FAN_SPACING = 0.78 * FAN_UNIT            # dot spacing inside the fan
FAN_INSIDE = 0.4                            # fraction of the fan that spills inward over the inner circle
def resample(px_, py_, valid, spacing, phase, min_dots=1):
    """Equal-arc-length dots along a polyline, breaking wherever valid is False."""
    out = []
    i = 0
    while i < len(px_):
        if not valid[i]: i += 1; continue
        j = i
        while j < len(px_) and valid[j]: j += 1
        x_, y_ = px_[i:j], py_[i:j]
        sl = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x_), np.diff(y_)))])
        t = np.arange(phase * spacing, sl[-1] + 1e-9, spacing)
        if len(t) < min_dots:                          # skip stray fragments of outer rows
            i = j; continue
        out.append(np.column_stack([np.interp(t, sl, x_), np.interp(t, sl, y_)]))
        i = j
    return np.vstack(out) if out else np.empty((0, 2))
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
    w_rel = np.clip((weight_w / weight_w.max() - FAN_THRESHOLD) / (1 - FAN_THRESHOLD), 0, 1)
    fan_w = FAN_MAX_WIDTH * w_rel ** FAN_TAPER
    fan_w = np.where(np.sin(ca) < 0, fan_w, 0.0)      # fan only on the left; the right stays a clean single line
    gapped = np.concatenate([[False], np.diff(ca) > np.radians(3)])
    contiguous = ~(gapped | np.roll(gapped, -1))       # break on both sides of a gap
    # 1) the carefully dotted inner line
    rx, ry = bmin * np.sin(ca), -bmin * np.cos(ca)
    # spacing grows steadily from the left end to the right end (ca runs right -> left)
    keep_pts = contiguous.copy()
    lx, ly = rx[keep_pts], ry[keep_pts]
    sl = np.concatenate([[0], np.cumsum(np.hypot(np.diff(lx), np.diff(ly)))])
    L = sl[-1]; t = [0.0]
    while True:
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
    b_in = bmin - FAN_INSIDE * fan_w
    span = fan_w
    dA = span * bmin * np.radians(0.5)                 # area of each 0.5-degree slice
    n_c = int(30 * dA.sum() / FAN_SPACING**2) + 1
    pick = rng.choice(len(ca), n_c, p=dA / dA.sum())
    c_a = ca[pick] + rng.uniform(-0.5, 0.5, n_c) * np.radians(0.5)
    c_b = b_in[pick] + rng.uniform(0, 1, n_c) * span[pick]
    cand_f = np.column_stack([c_b * np.sin(c_a), -c_b * np.cos(c_a)])
    fixed = np.vstack([dots, line]); fixed_tree = cKDTree(fixed)
    ok = fixed_tree.query(cand_f)[0] > FAN_SPACING
    cand_f = cand_f[ok]
    fan = []
    for p in cand_f[rng.permutation(len(cand_f))]:
        if not fan or np.min(np.hypot(*(np.array(fan) - p).T)) > FAN_SPACING:
            fan.append(p)
    fan = np.array(fan).reshape(-1, 2)
    dots = np.vstack([dots, line, fan])
    print("ring line dots:", len(line), " fan dots:", len(fan))

# 3) Blobby ink dots: each dot is a main blob plus a small offset lobe, with
#    size varying from dot to dot, on dark charcoal paper rather than pure black.
PAPER, INK = '#221f1b', '#ece7cf'      # warm dark gray paper, cream ink
RES_SCALE = 2                                          # output resolution multiplier (1 = 2800x1400 px)
BASE = 1.5                                             # typical dot diameter in points
FIG_W, FIG_H = 14, 7                                   # figure size in inches (2*VIEW x VIEW in M)
size = BASE * rng.lognormal(0, 0.10, len(dots))
ang = rng.uniform(0, 2 * np.pi, len(dots)); off = rng.uniform(0.08, 0.25, len(dots))
pt_to_M = 2*VIEW / (FIG_W * 72)                        # figure is 14 in wide for 2*VIEW M
lobe = dots + np.column_stack([np.cos(ang), np.sin(ang)]) * (off * size * pt_to_M)[:, None]
LOBE_SCALE = 0.7                                       # lobe diameter relative to its dot
INK_BLUR = 0.35                                        # blur radius in pixels at RES_SCALE = 1

# --- 5. Paper and photographic texture ---
# The original isn't flat black: paper fibers, ink wicking and the photographic
# reproduction leave a fine grain, a patchy mid-scale mottle and slow tonal drift,
# all in a warm dark gray. Applied as a final pass; dot placement is untouched.
TEXTURE = True
GRAIN, MOTTLE, DRIFT = 3.5, 2.1, 2.8     # strength of each layer, in 0-255 brightness levels
EDGE_ROUGHNESS = 0.07                    # how much fiber texture eats into dot edges
PAPER_RGB = np.array([int(PAPER[i:i+2], 16) for i in (1, 3, 5)], float)
INK_RGB = np.array([int(INK[i:i+2], 16) for i in (1, 3, 5)], float)

def texture_layers(h_, w_, res_scale):
    """Fixed noise fields (grain, mottle, drift) for an h_ x w_ image."""
    from PIL import Image
    trng = np.random.default_rng(7)
    def noise(sigma):
        # pixel sigmas are given at RES_SCALE = 1; large ones are made at low resolution and upsampled
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
    def __init__(self, h_, w_, res_scale, strength=1.0):
        # strength scales every texture effect (grain, mottle, tonal drift, rough edges)
        grain, mottle, drift = texture_layers(h_, w_, res_scale)
        grain, mottle, drift = strength * grain, strength * mottle, strength * drift
        self.rough = EDGE_ROUGHNESS * grain * 4
        self.paper = PAPER_RGB[None, None, :] + (GRAIN * grain + MOTTLE * mottle + DRIFT * drift)[..., None]
        self.ink = INK_RGB[None, None, :] * (1 + 0.01 * grain + 0.02 * drift)[..., None]
    def __call__(self, m):
        return self.patch(m, (slice(None), slice(None)))
    def patch(self, m, region):
        """Texture just the pixels in region (a pair of slices); m covers that region."""
        # rough, fibrous dot edges: perturb the ink mask most where it is partial (edges)
        m = np.clip(m + self.rough[region] * m * (1 - m), 0, 1)
        out = self.paper[region] * (1 - m[..., None]) + self.ink[region] * m[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

def ink_coverage(img):
    """How much ink each pixel of an RGB image holds, 0 = bare paper, 1 = solid dot."""
    d = INK_RGB - PAPER_RGB
    return np.clip(((img - PAPER_RGB) @ d) / (d @ d), 0, 1)

if __name__ == '__main__':
    from PIL import Image, ImageFilter
    DPI = 200 * RES_SCALE
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(PAPER)
    ax.scatter(dots[:, 0], dots[:, 1], s=size**2, c=INK, linewidths=0)
    ax.scatter(lobe[:, 0], lobe[:, 1], s=(LOBE_SCALE * size)**2, c=INK, linewidths=0)
    ax.set_xlim(-VIEW, VIEW); ax.set_ylim(-VIEW/2, VIEW/2); ax.axis('off')
    fig.savefig('luminet_recreation.png', dpi=DPI, facecolor=PAPER)
    # Soften edges slightly, like ink soaking into paper
    Image.open('luminet_recreation.png').filter(ImageFilter.GaussianBlur(INK_BLUR * RES_SCALE)).save('luminet_recreation.png')
    if TEXTURE:
        img = np.asarray(Image.open('luminet_recreation.png').convert('RGB')).astype(np.float32)
        m = ink_coverage(img)
        Image.fromarray(Texturer(*m.shape, RES_SCALE)(m)).save('luminet_recreation.png')
    print("candidates:", len(pts), " dots drawn:", len(dots))
