"""
Recreating J.-P. Luminet's 1979 black hole image
("Image of a spherical black hole with thin accretion disk", A&A 75, 228).

The rendering engine is blackhole.py; this script pins the exact parameters of
the recreation (blackhole.Params' defaults, the 'luminet1979' preset) and writes
luminet_recreation.png. To draw other black holes, use blackhole.py.
"""
from blackhole import Params, make_drawing, render_still

LUMINET_1979 = Params()      # observer 80 deg from the disk axis, disk from 6M to 400M, ...
RES_SCALE = 2                # 5600 x 2800 px

if __name__ == '__main__':
    d = make_drawing(LUMINET_1979)
    render_still(d, 'luminet_recreation.png', RES_SCALE)
    print("dots drawn:", len(d.dots))
