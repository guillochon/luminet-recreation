# Recreating Luminet's 1979 black hole image

In 1979, Jean-Pierre Luminet published the first computed image of a black hole surrounded by a glowing accretion disk ("Image of a Spherical Black Hole with Thin Accretion Disk", *Astronomy & Astrophysics* 75, 228–235). He computed the light paths on an IBM 7040 at the Paris Observatory, then drew the final image by hand, dot by dot, in India ink. Forty years later it turned out to look strikingly like the Event Horizon Telescope's real image of M87*.

This project recreates that image. The physics follows the paper; the rendering imitates his hand-inked stippling, tuned against closeups of the original drawing.

![Recreation](luminet_recreation.png)

## Running it

```bash
pip install numpy scipy matplotlib pillow scikit-image
python luminet_1979.py
```

This writes `luminet_recreation.png` (5600 × 2800 px by default) to the current directory. It may take a minute or two.

### Animation

```bash
python luminet_animation.py
```

This writes `luminet_animation.mp4`, a 60 fps video of the drawing being inked, at half the still's resolution (2800 × 1400 px). It needs `ffmpeg` on your PATH. At the default of 3 dots per frame, the 24,454 dots take about 2¼ minutes. Run `python luminet_animation.py --help` for all options, such as `--dots-per-frame`, `--dry-seconds` and `--texture`.

- **Drying ink.** Each drop lands wet: about 15% smaller than its final size and domed like a droplet, lit from the upper left with a small glint. Over 5 seconds it flattens and spreads to its final size. When it's dry it is exactly the dot in the still.
- **Lighter paper texture.** The video uses the paper grain and tone variation at half the still's strength (`--texture 0.5`). Use `--texture 1` to match the still.

- **Drawing order.** Dots go down from the inside out, along a short pen path. The drawing is split into thin bands, each about two dot spacings wide. Near the hole the bands are circles; farther out they stretch to the frame's 2:1 shape. The pen sweeps around each band, alternating direction, so every band starts where the previous one ended. A traveling-salesman improvement step (2-opt) then shortens the route. The typical hop between consecutive dots is 0.15 M, close to the 0.12 M spacing between neighboring dots.
- **Same image as the still.** Each frame is built the same way as the still: same dots, same blur, same paper texture. With `--texture 1`, the last frame matches a half-resolution render of the still.
- **Video compression.** Thousands of tiny dots on grainy paper are hard for a video encoder. Three settings deal with that. First, the paper texture stays fixed from frame to frame, so each frame differs from the last only around the drops that are still drying, which keeps the video small. Second, the H.264 encoder runs at high quality (CRF 12) with `-tune grain`, so it doesn't smooth away the grain or the dot edges. Third, the file uses an explicit, tagged BT.709 color conversion, so players don't shift the colors. Use `--lossless` for an exact RGB master (`.mkv`) to re-encode from, or `--pix-fmt yuv444p` for full color resolution.

## How it works

### 1. Physics (from the paper)

- **Light bending.** Photon paths around a non-spinning (Schwarzschild) black hole are integrated numerically for a range of impact parameters, using the orbit equation u″ = −u + 3Mu².
- **Geometry.** The observer sits 10° above the disk plane (80° from its axis), as in the paper's Fig. 11. Each pixel's ray is traced to where it crosses the disk, giving the **direct (primary)** image and the **secondary** image: light that went partway around the hole and forms the thin inner ring.
- **Disk brightness.** The disk runs from the innermost stable orbit (6M) outward and glows according to the Page–Thorne formula (Eq. 15).
- **Redshift.** Observed brightness is the intrinsic brightness divided by (1+z)⁴, where 1+z combines gravitational redshift and the Doppler shift from the disk's rotation (Eq. 19). This makes the approaching (left) side far brighter than the receding (right) side.

**A note on Eq. 15.** The paper prints the coefficient of the log term as √3/3. The correct Page–Thorne value is √3/2, and only √3/2 reproduces the paper's own stated numbers: peak brightness at r ≈ 9.55M and a normalized peak of 1.146×10⁻⁴. The script uses √3/2. With it, the maximum observed brightness comes out at 2.81 versus 2.62 in the paper's Fig. 10, a gap plausibly due to his coarser grid.

### 2. Rendering (imitating the hand-inked drawing)

- **Even dot spacing.** Dots are placed by Poisson-disk sampling: each dot keeps a minimum distance from its neighbors, and that distance shrinks where the disk is brighter. Brightness becomes dot spacing, much as a person inking by hand would build up tone.
- **Clean inner edge.** The first rows of dots along the shadow's edge are placed in regular rows, loosening into random placement a few rows into the disk.
- **Inner ring.** The secondary image is drawn as a carefully spaced dotted line along its inner edge. The spacing doubles from left to right, and the line flares into a fan at the bright lower-left end, with a short two-dot stretch at the lower right.
- **Isoradial banding.** A very faint modulation along curves of constant disk radius echoes how Luminet's data came as isoradial curves.
- **Ink and paper.** Dots are slightly irregular cream blobs on warm dark-gray paper, finished with fine grain, mid-scale mottling and slow tonal drift.

## Faithful vs. tuned

Some choices follow the paper; others were tuned by eye to match the original drawing, because the paper doesn't say how brightness was turned into ink.

| Faithful to the paper | Tuned to match the drawing |
|---|---|
| Viewing angle, disk inner edge, light bending | How brightness maps to dot density (`DOT_GAMMA`, `DOT_GAIN`, `DOT_CAP`) |
| Page–Thorne brightness (with the √3/2 correction) | Sharp inner disk edge (`SHARP_INNER_EDGE`), which departs from strict physics |
| Gravitational + Doppler redshift, (1+z)⁴ dimming | The inner ring's dot spacing and fan shapes |
| Primary and secondary images, opaque disk | Background speckle, banding, paper texture, dot size |

The paper says Luminet estimated the inner ring's brightness with fixed values (r = 9.55M, b = 5.25M) rather than computing it. Setting `LUMINET_GHOST = True` uses that approximation; the default draws the fully computed ring.

## Main settings

All settings are constants near where they're used in `luminet_1979.py`.

| Setting | What it controls |
|---|---|
| `VIEW` | Half-width of the frame, in units of the black hole's mass |
| `RES_SCALE` | Output resolution multiplier; blur and texture scale with it |
| `SPACING_AT_PEAK` | Dot spacing in the brightest region (smaller = more dots overall) |
| `DOT_GAMMA` | Values above 1 thin the dim right and outer disk |
| `DOT_GAIN` | Pushes mid-bright regions toward full density (the white "claw") |
| `DOT_CAP` | How packed the brightest, saturated region is |
| `BACKGROUND_DENSITY` | Faint speckle in the darkest areas |
| `EDGE_SOFTNESS` | How quickly the disk fades in from the shadow edge |
| `EDGE_ROWS` | Rows of regular dots along the shadow edge before random placement |
| `RING_SPACING_LEFT`, `RING_SPACING_RIGHT` | Inner-ring dot spacing at each end |
| `FAN_MAX_WIDTH`, `FAN_TAPER`, `FAN_THRESHOLD`, `FAN_SPACING`, `FAN_INSIDE` | Shape and density of the left fan |
| `RIGHT_FAN_DOTS` | Length of the two-dot stretch on the right |
| `RING_STEP`, `BAND_DEPTH` | Spacing and strength of isoradial banding |
| `BASE`, `INK_BLUR` | Dot size and edge softness |
| `TEXTURE`, `GRAIN`, `MOTTLE`, `DRIFT`, `EDGE_ROUGHNESS` | Paper and photographic texture |

## References

- J.-P. Luminet, "Image of a Spherical Black Hole with Thin Accretion Disk", *Astronomy & Astrophysics* 75, 228–235 (1979). [NASA ADS](https://ui.adsabs.harvard.edu/abs/1979A&A....75..228L/abstract)
- D. N. Page and K. S. Thorne, "Disk-Accretion onto a Black Hole. Time-Averaged Structure of Accretion Disk", *Astrophysical Journal* 191, 499 (1974).
- The original image on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BH-JPL-A%26A1979.jpg) (CC BY-SA 4.0; credit Jean-Pierre Luminet / CNRS Photothèque).

## License

MIT. See [LICENSE](LICENSE).
