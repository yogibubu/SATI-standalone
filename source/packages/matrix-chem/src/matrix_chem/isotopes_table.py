"""

Isotopic table for ORACLE.

Derived from Wapstra & Bos, Atomic and Nuclear Data Tables, 1977).

All data are suitable for high–resolution molecular spectroscopy.

NOTE that  Atomic masses are ATOMIC (not nuclear) masses

"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Isotope:
    Z: int               # atomic number
    A: int               # mass number
    mass: float          # atomic mass (amu)
    spin2: int           # 2 * nuclear spin
    g_factor: float      # nuclear magnetic moment
    quadrupole: float    # nuclear quadrupole moment (fm^2); convert to barn by /100


# ----------------------------------------------------------------------
# Isotopic data
# The FIRST isotope in the list is the DEFAULT one (most abundant /
# spectroscopically standard).
# ----------------------------------------------------------------------

ISOTOPES: Dict[int, List[Isotope]] = {

    # -------------------------
    # Z = 1  Hydrogen
    # -------------------------
    1: [
        Isotope(Z=1, A=1, mass=1.007825037, spin2=1, g_factor=2.792846000, quadrupole=0.0),
        Isotope(Z=1, A=2, mass=2.014101787, spin2=2, g_factor=0.857438000, quadrupole=0.286),
        Isotope(Z=1, A=3, mass=3.016049286, spin2=1, g_factor=2.978960000, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 2  Helium
    # -------------------------
    2: [
        Isotope(Z=2, A=4, mass=4.002603250, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=2, A=3, mass=3.016029297, spin2=1, g_factor=-2.127620000, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 3  Lithium
    # -------------------------
    3: [
        Isotope(Z=3, A=7, mass=7.016004500, spin2=3, g_factor=3.256424000, quadrupole=-4.010),
        Isotope(Z=3, A=6, mass=6.015123200, spin2=2, g_factor=0.822047000, quadrupole=-0.0808),
    ],

    # -------------------------
    # Z = 4  Beryllium
    # -------------------------
    4: [
        Isotope(Z=4, A=9, mass=9.012182500, spin2=3, g_factor=-1.177900000, quadrupole=5.288),
    ],

    # -------------------------
    # Z = 5  Boron
    # -------------------------
    5: [
        Isotope(Z=5, A=11, mass=11.009305300, spin2=3, g_factor=2.688637000, quadrupole=4.059),
        Isotope(Z=5, A=10, mass=10.012938000, spin2=6, g_factor=1.800650000, quadrupole=8.459),
    ],

    # -------------------------
    # Z = 6  Carbon
    # -------------------------
    6: [
        Isotope(Z=6, A=12, mass=12.000000000, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=6, A=13, mass=13.003354839, spin2=1, g_factor=0.702411000, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 7  Nitrogen
    # -------------------------
    7: [
        Isotope(Z=7, A=14, mass=14.003074008, spin2=2, g_factor=0.403761000, quadrupole=2.044),
        Isotope(Z=7, A=15, mass=15.000108978, spin2=1, g_factor=-0.283190000, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 8  Oxygen
    # -------------------------
    8: [
        Isotope(Z=8, A=16, mass=15.994914640, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=8, A=18, mass=17.999159390, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=8, A=17, mass=16.999130600, spin2=5, g_factor=-1.893800000, quadrupole=-2.558),
    ],

    # -------------------------
    # Z = 9  Fluorine
    # -------------------------
    9: [
        Isotope(Z=9, A=19, mass=18.998403250, spin2=1, g_factor=2.628867000, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 10  Neon
    # -------------------------
    10: [
        Isotope(Z=10, A=20, mass=19.992439100, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=10, A=22, mass=21.991383700, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=10, A=21, mass=20.993845300, spin2=3, g_factor=-0.661800000, quadrupole=10.155),
    ],

    # -------------------------
    # Z = 11  Sodium
    # -------------------------
    11: [
        Isotope(Z=11, A=23, mass=22.9897697, spin2=3, g_factor=2.21752, quadrupole=10.4),
    ],

    # -------------------------
    # Z = 12  Magnesium
    # -------------------------
    12: [
        Isotope(Z=12, A=24, mass=23.9850450, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=12, A=26, mass=25.9825954, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=12, A=25, mass=24.9858392, spin2=5, g_factor=-0.85546, quadrupole=19.94),
    ],

    # -------------------------
    # Z = 13  Aluminium
    # -------------------------
    13: [
        Isotope(Z=13, A=27, mass=26.9815413, spin2=5, g_factor=3.641504, quadrupole=14.66),
    ],

    # -------------------------
    # Z = 14  Silicon
    # -------------------------
    14: [
        Isotope(Z=14, A=28, mass=27.9769284, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=14, A=29, mass=28.9764964, spin2=1, g_factor=-0.55529, quadrupole=0.0),
        Isotope(Z=14, A=30, mass=29.9737717, spin2=0, g_factor=0.0, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 15  Phosphorus
    # -------------------------
    15: [
        Isotope(Z=15, A=31, mass=30.9737634, spin2=1, g_factor=1.1316, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 16  Sulfur
    # -------------------------
    16: [
        Isotope(Z=16, A=32, mass=31.9720718, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=16, A=34, mass=33.96786774, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=16, A=33, mass=32.9714591, spin2=3, g_factor=0.643821, quadrupole=-6.78),
        Isotope(Z=16, A=36, mass=35.9670790, spin2=0, g_factor=0.0, quadrupole=0.0),
    ],

    # -------------------------
    # Z = 17  Chlorine
    # -------------------------
    17: [
        Isotope(Z=17, A=35, mass=34.968852729, spin2=3, g_factor=0.821874, quadrupole=-8.165),
        Isotope(Z=17, A=37, mass=36.965902624, spin2=3, g_factor=0.684123, quadrupole=-6.435),
    ],

    # -------------------------
    # Z = 18  Argon
    # -------------------------
    18: [
        Isotope(Z=18, A=40, mass=39.9623831, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=18, A=36, mass=35.967545605, spin2=0, g_factor=0.0, quadrupole=0.0),
        Isotope(Z=18, A=38, mass=37.9627322, spin2=0, g_factor=0.0, quadrupole=0.0),
    ],
# -------------------------
# Z = 19  Potassium
# -------------------------
19: [
    Isotope(Z=19, A=39, mass=38.9637079, spin2=3, g_factor=0.391466, quadrupole=5.85),
    Isotope(Z=19, A=41, mass=40.9618254, spin2=3, g_factor=0.21487, quadrupole=7.11),
    Isotope(Z=19, A=40, mass=39.9639988, spin2=8, g_factor=-1.2981, quadrupole=-7.3),
],

# -------------------------
# Z = 20  Calcium
# -------------------------
20: [
    Isotope(Z=20, A=40, mass=39.9625907, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=20, A=44, mass=43.9554848, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=20, A=42, mass=41.9586218, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=20, A=48, mass=47.952532, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 21  Scandium
# -------------------------
21: [
    Isotope(Z=21, A=45, mass=44.9559136, spin2=7, g_factor=4.756483, quadrupole=-22.0),
],

# -------------------------
# Z = 22  Titanium
# -------------------------
22: [
    Isotope(Z=22, A=48, mass=47.9479467, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=22, A=46, mass=45.9526327, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=22, A=47, mass=46.9517649, spin2=5, g_factor=-0.78848, quadrupole=30.2),
    Isotope(Z=22, A=49, mass=48.9478705, spin2=7, g_factor=-1.10417, quadrupole=24.7),
],

# -------------------------
# Z = 23  Vanadium
# -------------------------
23: [
    Isotope(Z=23, A=51, mass=50.9439625, spin2=7, g_factor=5.1514, quadrupole=-5.2),
    Isotope(Z=23, A=50, mass=49.9471613, spin2=12, g_factor=3.34745, quadrupole=21.0),
],

# -------------------------
# Z = 24  Chromium
# -------------------------
24: [
    Isotope(Z=24, A=52, mass=51.9405097, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=24, A=53, mass=52.940651, spin2=3, g_factor=-0.47454, quadrupole=-15.0),
    Isotope(Z=24, A=50, mass=49.9460463, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=24, A=54, mass=53.9388822, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 25  Manganese
# -------------------------
25: [
    Isotope(Z=25, A=55, mass=54.9380463, spin2=5, g_factor=3.4532, quadrupole=33.0),
],

# -------------------------
# Z = 26  Iron
# -------------------------
26: [
    Isotope(Z=26, A=56, mass=55.9349393, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=26, A=54, mass=53.9396121, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=26, A=57, mass=56.9353957, spin2=1, g_factor=0.090623, quadrupole=16.0),
    Isotope(Z=26, A=58, mass=57.9332778, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 27  Cobalt
# -------------------------
27: [
    Isotope(Z=27, A=59, mass=58.9331978, spin2=7, g_factor=4.627, quadrupole=42.0),
],

# -------------------------
# Z = 28  Nickel
# -------------------------
28: [
    Isotope(Z=28, A=58, mass=57.9353471, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=28, A=60, mass=59.930789, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=28, A=62, mass=61.9283464, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=28, A=61, mass=60.9310586, spin2=3, g_factor=-0.75002, quadrupole=16.2),
],

# -------------------------
# Z = 29  Copper
# -------------------------
29: [
    Isotope(Z=29, A=63, mass=62.9295992, spin2=3, g_factor=2.2233, quadrupole=-22.0),
    Isotope(Z=29, A=65, mass=64.9277924, spin2=3, g_factor=2.3817, quadrupole=-20.4),
],

# -------------------------
# Z = 30  Zinc
# -------------------------
30: [
    Isotope(Z=30, A=64, mass=63.9291454, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=30, A=66, mass=65.9260352, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=30, A=68, mass=67.9248458, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=30, A=67, mass=66.9271289, spin2=5, g_factor=0.875479, quadrupole=15.0),
],

# -------------------------
# Z = 31  Gallium
# -------------------------
31: [
    Isotope(Z=31, A=69, mass=68.9255809, spin2=3, g_factor=2.01659, quadrupole=17.1),
    Isotope(Z=31, A=71, mass=70.9247006, spin2=3, g_factor=2.56227, quadrupole=10.7),
],

# -------------------------
# Z = 32  Germanium
# -------------------------
32: [
    Isotope(Z=32, A=74, mass=73.9211788, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=32, A=72, mass=71.9220800, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=32, A=70, mass=69.9242498, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=32, A=73, mass=72.9234639, spin2=9, g_factor=-0.87947, quadrupole=-19.6),
],

# -------------------------
# Z = 33  Arsenic
# -------------------------
33: [
    Isotope(Z=33, A=75, mass=74.9215955, spin2=3, g_factor=1.43947, quadrupole=31.4),
],

# -------------------------
# Z = 34  Selenium
# -------------------------
34: [
    Isotope(Z=34, A=80, mass=79.9165205, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=34, A=78, mass=77.9173040, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=34, A=82, mass=81.9167090, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=34, A=76, mass=75.9192066, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 35  Bromine
# -------------------------
35: [
    Isotope(Z=35, A=79, mass=78.9183361, spin2=3, g_factor=2.106399, quadrupole=31.3),
    Isotope(Z=35, A=81, mass=80.9162900, spin2=3, g_factor=2.27056, quadrupole=26.2),
],

# -------------------------
# Z = 36  Krypton
# -------------------------
36: [
    Isotope(Z=36, A=84, mass=83.9115064, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=36, A=86, mass=85.9106140, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=36, A=82, mass=81.9134830, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=36, A=83, mass=82.9141340, spin2=9, g_factor=-0.97067, quadrupole=25.9),
],

# -------------------------
# Z = 37  Rubidium
# -------------------------
37: [
    Isotope(Z=37, A=85, mass=84.9117000, spin2=5, g_factor=1.35303, quadrupole=27.6),
    Isotope(Z=37, A=87, mass=86.9091870, spin2=3, g_factor=2.75124, quadrupole=13.35),
],

# -------------------------
# Z = 38  Strontium
# -------------------------
38: [
    Isotope(Z=38, A=88, mass=87.9056000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=38, A=84, mass=83.9134000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=38, A=86, mass=85.9094000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=38, A=87, mass=86.9089000, spin2=9, g_factor=-1.09283, quadrupole=33.5),
],

# -------------------------
# Z = 39  Yttrium
# -------------------------
39: [
    Isotope(Z=39, A=89, mass=88.9054000, spin2=1, g_factor=-0.13742, quadrupole=0.0),
],

# -------------------------
# Z = 40  Zirconium
# -------------------------
40: [
    Isotope(Z=40, A=90, mass=89.9043000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=40, A=91, mass=90.9053000, spin2=5, g_factor=-1.30362, quadrupole=-17.6),
    Isotope(Z=40, A=92, mass=91.9046000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=40, A=94, mass=93.9061000, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 41  Niobium
# -------------------------
41: [
    Isotope(Z=41, A=93, mass=92.9060000, spin2=9, g_factor=6.1705, quadrupole=-32.0),
],

# -------------------------
# Z = 42  Molybdenum
# -------------------------
42: [
    Isotope(Z=42, A=98, mass=97.9055000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=42, A=92, mass=91.9063000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=42, A=95, mass=94.9058400, spin2=5, g_factor=-0.9142, quadrupole=-2.2),
],

# -------------------------
# Z = 43  Technetium
# -------------------------
43: [
    Isotope(Z=43, A=99, mass=98.9063000, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 44  Ruthenium
# -------------------------
44: [
    Isotope(Z=44, A=102, mass=101.9037000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=44, A=99,  mass=98.9061000, spin2=5, g_factor=-0.6413, quadrupole=7.9),
    Isotope(Z=44, A=100, mass=99.9030000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=44, A=104, mass=103.9055000, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 45  Rhodium
# -------------------------
45: [
    Isotope(Z=45, A=103, mass=102.9048000, spin2=1, g_factor=-0.0884, quadrupole=0.0),
],

# -------------------------
# Z = 46  Palladium
# -------------------------
46: [
    Isotope(Z=46, A=106, mass=105.9032000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=46, A=104, mass=103.9036000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=46, A=105, mass=104.9046000, spin2=5, g_factor=-0.6420, quadrupole=66.0),
    Isotope(Z=46, A=108, mass=107.9038900, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 47  Silver
# -------------------------
47: [
    Isotope(Z=47, A=107, mass=106.9050900, spin2=1, g_factor=-0.11357, quadrupole=0.0),
    Isotope(Z=47, A=109, mass=108.9047000, spin2=1, g_factor=-0.13069, quadrupole=0.0),
],

# -------------------------
# Z = 48  Cadmium
# -------------------------
48: [
    Isotope(Z=48, A=114, mass=113.9036000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=48, A=110, mass=109.9030000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=48, A=111, mass=110.9042000, spin2=1, g_factor=-0.59489, quadrupole=0.0),
    Isotope(Z=48, A=112, mass=111.9028000, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 49  Indium
# -------------------------
49: [
    Isotope(Z=49, A=115, mass=114.9041000, spin2=9, g_factor=5.5408, quadrupole=81.0),
    Isotope(Z=49, A=113, mass=112.9043000, spin2=9, g_factor=5.5289, quadrupole=79.9),
],

# -------------------------
# Z = 50  Tin
# -------------------------
50: [
    Isotope(Z=50, A=118, mass=117.9018000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=50, A=116, mass=115.9021000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=50, A=117, mass=116.9031000, spin2=1, g_factor=-1.00105, quadrupole=0.0),
    Isotope(Z=50, A=119, mass=118.9034000, spin2=1, g_factor=-1.04729, quadrupole=-12.8),
],

# -------------------------
# Z = 51  Antimony
# -------------------------
51: [
    Isotope(Z=51, A=121, mass=120.9038000, spin2=5, g_factor=3.3634, quadrupole=-36.0),
    Isotope(Z=51, A=123, mass=122.9041000, spin2=7, g_factor=2.5498, quadrupole=-49.0),
],

# -------------------------
# Z = 52  Tellurium
# -------------------------
52: [
    Isotope(Z=52, A=130, mass=129.9067000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=52, A=128, mass=127.9045000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=52, A=125, mass=124.9044300, spin2=1, g_factor=-0.8870, quadrupole=0.0),
],

# -------------------------
# Z = 53  Iodine
# -------------------------
53: [
    Isotope(Z=53, A=127, mass=126.9045000, spin2=5, g_factor=2.8133, quadrupole=-71.0),
],

# -------------------------
# Z = 54  Xenon
# -------------------------
54: [
    Isotope(Z=54, A=132, mass=131.9041000, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=54, A=129, mass=128.9047800, spin2=1, g_factor=-0.77797, quadrupole=0.0),
    Isotope(Z=54, A=131, mass=130.9050800, spin2=3, g_factor=0.6910, quadrupole=0.0),
],

# -------------------------
# Z = 55  Cesium
# -------------------------
55: [
    Isotope(Z=55, A=133, mass=132.905429, spin2=7, g_factor=2.582025, quadrupole=-0.003),
],

# -------------------------
# Z = 56  Barium
# -------------------------
56: [
    Isotope(Z=56, A=138, mass=137.905232, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=56, A=137, mass=136.905821, spin2=3, g_factor=0.937365, quadrupole=0.245),
    Isotope(Z=56, A=135, mass=134.905683, spin2=3, g_factor=0.838195, quadrupole=0.160),
],

# -------------------------
# Z = 57  Lanthanum
# -------------------------
57: [
    Isotope(Z=57, A=139, mass=138.906348, spin2=7, g_factor=2.783045, quadrupole=0.20),
    Isotope(Z=57, A=138, mass=137.907107, spin2=5, g_factor=3.641, quadrupole=0.12),
],

# -------------------------
# Z = 58  Cerium
# -------------------------
58: [
    Isotope(Z=58, A=140, mass=139.905435, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=58, A=142, mass=141.909241, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 59  Praseodymium
# -------------------------
59: [
    Isotope(Z=59, A=141, mass=140.907648, spin2=5, g_factor=4.275, quadrupole=-0.058),
],

# -------------------------
# Z = 60  Neodymium
# -------------------------
60: [
    Isotope(Z=60, A=144, mass=143.910083, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=60, A=143, mass=142.909816, spin2=7, g_factor=-1.208, quadrupole=-0.30),
],

# -------------------------
# Z = 61  Promethium
# -------------------------
61: [
    Isotope(Z=61, A=145, mass=144.912749, spin2=5, g_factor=2.9, quadrupole=0.0),
],

# -------------------------
# Z = 62  Samarium
# -------------------------
62: [
    Isotope(Z=62, A=152, mass=151.919732, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=62, A=154, mass=153.922209, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 63  Europium
# -------------------------
63: [
    Isotope(Z=63, A=153, mass=152.921238, spin2=5, g_factor=1.533, quadrupole=2.41),
    Isotope(Z=63, A=151, mass=150.919846, spin2=5, g_factor=3.471, quadrupole=0.90),
],

# -------------------------
# Z = 64  Gadolinium
# -------------------------
64: [
    Isotope(Z=64, A=158, mass=157.924103, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=64, A=160, mass=159.927054, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 65  Terbium
# -------------------------
65: [
    Isotope(Z=65, A=159, mass=158.925346, spin2=3, g_factor=2.014, quadrupole=1.43),
],

# -------------------------
# Z = 66  Dysprosium
# -------------------------
66: [
    Isotope(Z=66, A=164, mass=163.929174, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=66, A=163, mass=162.928733, spin2=5, g_factor=-0.265, quadrupole=2.51),
],

# -------------------------
# Z = 67  Holmium
# -------------------------
67: [
    Isotope(Z=67, A=165, mass=164.930319, spin2=7, g_factor=4.132, quadrupole=3.58),
],

# -------------------------
# Z = 68  Erbium
# -------------------------
68: [
    Isotope(Z=68, A=166, mass=165.930293, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=68, A=168, mass=167.932370, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 69  Thulium
# -------------------------
69: [
    Isotope(Z=69, A=169, mass=168.934212, spin2=1, g_factor=-1.04, quadrupole=0.0),
],

# -------------------------
# Z = 70  Ytterbium
# -------------------------
70: [
    Isotope(Z=70, A=174, mass=173.938873, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=70, A=176, mass=175.942576, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 71  Lutetium
# -------------------------
71: [
    Isotope(Z=71, A=175, mass=174.940772, spin2=7, g_factor=2.232, quadrupole=3.49),
    Isotope(Z=71, A=176, mass=175.942686, spin2=14, g_factor=3.64, quadrupole=4.92),
],

# -------------------------
# Z = 72  Hafnium
# -------------------------
72: [
    Isotope(Z=72, A=178, mass=177.9436977, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=72, A=176, mass=175.9414018, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=72, A=177, mass=176.9432207, spin2=7, g_factor=0.7936, quadrupole=3.7),
    Isotope(Z=72, A=179, mass=178.9458151, spin2=9, g_factor=0.7936, quadrupole=3.7),
    Isotope(Z=72, A=180, mass=179.9465488, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 73  Tantalum
# -------------------------
73: [
    Isotope(Z=73, A=181, mass=180.9479958, spin2=7, g_factor=2.3705, quadrupole=3.17),
],

# -------------------------
# Z = 74  Tungsten
# -------------------------
74: [
    Isotope(Z=74, A=184, mass=183.9509312, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=74, A=186, mass=185.9543641, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=74, A=182, mass=181.9482042, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=74, A=183, mass=182.9502230, spin2=1, g_factor=0.1178, quadrupole=0.0),
],

# -------------------------
# Z = 75  Rhenium
# -------------------------
75: [
    Isotope(Z=75, A=187, mass=186.9557501, spin2=5, g_factor=3.1871, quadrupole=2.07),
    Isotope(Z=75, A=185, mass=184.9529550, spin2=5, g_factor=3.1871, quadrupole=2.07),
],

# -------------------------
# Z = 76  Osmium
# -------------------------
76: [
    Isotope(Z=76, A=192, mass=191.9614770, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=76, A=190, mass=189.9584450, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=76, A=188, mass=187.9558382, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=76, A=189, mass=188.9581442, spin2=3, g_factor=0.6530, quadrupole=0.0),
],

# -------------------------
# Z = 77  Iridium
# -------------------------
77: [
    Isotope(Z=77, A=193, mass=192.9629216, spin2=3, g_factor=0.1641, quadrupole=0.75),
    Isotope(Z=77, A=191, mass=190.9605940, spin2=3, g_factor=0.1641, quadrupole=0.75),
],

# -------------------------
# Z = 78  Platinum
# -------------------------
78: [
    Isotope(Z=78, A=195, mass=194.9647917, spin2=1, g_factor=0.6090, quadrupole=0.0),
    Isotope(Z=78, A=194, mass=193.9626809, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=78, A=196, mass=195.9649521, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=78, A=198, mass=197.9678949, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 79  Gold
# -------------------------
79: [
    Isotope(Z=79, A=197, mass=196.9665688, spin2=3, g_factor=0.0970, quadrupole=0.547),
],

# -------------------------
# Z = 80  Mercury
# -------------------------
80: [
    Isotope(Z=80, A=202, mass=201.9706434, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=80, A=200, mass=199.9683269, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=80, A=199, mass=198.9682810, spin2=1, g_factor=0.5059, quadrupole=0.0),
    Isotope(Z=80, A=201, mass=200.9703028, spin2=3, g_factor=-0.5602, quadrupole=0.0),
],

# -------------------------
# Z = 81  Thallium
# -------------------------
81: [
    Isotope(Z=81, A=205, mass=204.9744278, spin2=1, g_factor=1.6382, quadrupole=0.24),
    Isotope(Z=81, A=203, mass=202.9723446, spin2=1, g_factor=1.6223, quadrupole=0.11),
],

# -------------------------
# Z = 82  Lead
# -------------------------
82: [
    Isotope(Z=82, A=208, mass=207.9766525, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=82, A=206, mass=205.9744653, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=82, A=207, mass=206.9758969, spin2=1, g_factor=0.5926, quadrupole=0.0),
],

# -------------------------
# Z = 83  Bismuth
# -------------------------
83: [
    Isotope(Z=83, A=209, mass=208.9803987, spin2=9, g_factor=4.1106, quadrupole=-0.52),
],

# -------------------------
# Z = 84  Polonium
# -------------------------
84: [
    Isotope(Z=84, A=209, mass=208.9824304, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 85  Astatine
# -------------------------
85: [
    Isotope(Z=85, A=210, mass=209.987148, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 86  Radon
# -------------------------
86: [
    Isotope(Z=86, A=222, mass=222.0175777, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 87  Francium
# -------------------------
87: [
    Isotope(Z=87, A=223, mass=223.0197359, spin2=3, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 88  Radium
# -------------------------
88: [
    Isotope(Z=88, A=226, mass=226.0254098, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 89  Actinium
# -------------------------
89: [
    Isotope(Z=89, A=227, mass=227.0277523, spin2=3, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 90  Thorium
# -------------------------
90: [
    Isotope(Z=90, A=232, mass=232.0380558, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 91  Protactinium
# -------------------------
91: [
    Isotope(Z=91, A=231, mass=231.0358842, spin2=3, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 92  Uranium
# -------------------------
92: [
    Isotope(Z=92, A=238, mass=238.0507884, spin2=0, g_factor=0.0, quadrupole=0.0),
    Isotope(Z=92, A=235, mass=235.0439299, spin2=7, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 93  Neptunium
# -------------------------
93: [
    Isotope(Z=93, A=237, mass=237.0481736, spin2=5, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 94  Plutonium
# -------------------------
94: [
    Isotope(Z=94, A=244, mass=244.0642053, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 95  Americium
# -------------------------
95: [
    Isotope(Z=95, A=243, mass=243.0613813, spin2=5, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 96  Curium
# -------------------------
96: [
    Isotope(Z=96, A=247, mass=247.0703541, spin2=9, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 97  Berkelium
# -------------------------
97: [
    Isotope(Z=97, A=247, mass=247.0703073, spin2=3, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 98  Californium
# -------------------------
98: [
    Isotope(Z=98, A=251, mass=251.0795886, spin2=1, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 99  Einsteinium
# -------------------------
99: [
    Isotope(Z=99, A=252, mass=252.082980, spin2=5, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 100  Fermium
# -------------------------
100: [
    Isotope(Z=100, A=257, mass=257.0951061, spin2=9, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 101  Mendelevium
# -------------------------
101: [
    Isotope(Z=101, A=258, mass=258.0984315, spin2=1, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 102  Nobelium
# -------------------------
102: [
    Isotope(Z=102, A=259, mass=259.10103, spin2=9, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 103  Lawrencium
# -------------------------
103: [
    Isotope(Z=103, A=262, mass=262.10961, spin2=1, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 104  Rutherfordium
# -------------------------
104: [
    Isotope(Z=104, A=267, mass=267.12179, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 105  Dubnium
# -------------------------
105: [
    Isotope(Z=105, A=268, mass=268.12567, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 106  Seaborgium
# -------------------------
106: [
    Isotope(Z=106, A=271, mass=271.13393, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 107  Bohrium
# -------------------------
107: [
    Isotope(Z=107, A=272, mass=272.13826, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 108  Hassium
# -------------------------
108: [
    Isotope(Z=108, A=270, mass=270.13429, spin2=0, g_factor=0.0, quadrupole=0.0),
],

# -------------------------
# Z = 109  Meitnerium
# -------------------------
109: [
    Isotope(Z=109, A=278, mass=278.15631, spin2=0, g_factor=0.0, quadrupole=0.0),
],

}

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_isotopes(Z: int) -> List[Isotope]:
    return ISOTOPES.get(Z, [])

def get_default_isotope(Z: int) -> Optional[Isotope]:
    """
    Return the default (most abundant) isotope:
    """
    lst = ISOTOPES.get(Z)
    if not lst:
        return None
    return lst[0]  # default first in the table

def get_isotope(Z: int, A: int) -> Optional[Isotope]:
    for iso in ISOTOPES.get(Z, []):
        if iso.A == A:
            return iso
    return None
