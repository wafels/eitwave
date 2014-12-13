#
# Demo information
#
import os
from copy import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from sunpy.net import hek
from sunpy.map import Map
import aware

info = {"previous1": {"tr": hek.attrs.Time('2011-10-01 08:56:00', '2011-10-01 10:17:00'),
                      "accum": 1,
                      "result": 1,
                      "lon_index": 30,
                      "listindex": [12, 14],
                      "pbd": "aia_lev1_211a_2011_10_01t08_54_00_62z_image_lev1_fits.fits"},
             "corpita_fig4": {"tr": hek.attrs.Time('2011-06-07 06:15:00', '2011-06-07 07:00:00'),
                               "accum": 2,
                               "result": 0},
             "corpita_fig6": {"tr": hek.attrs.Time('2011-02-08 21:10:00', '2011-02-08 21:21:00'),
                               "accum": 1},
             "corpita_fig7": {"tr": hek.attrs.Time('2011-02-13 17:32:48', '2011-02-13 17:48:48'),
                               "accum": 2,
                               "result": 0,
                               "listindex": [20, 22],
                               "pbd": "aia_lev1_211a_2011_02_13t17_30_48_62z_image_lev1_fits.fits"},
             "corpita_fig8a": {"tr": hek.attrs.Time('2011-02-15 01:48:00', '2011-02-15 02:14:24'),
                               "accum": 3,
                               "result": 0,
                               "lon_index": 23,
                               "listindex": [30, 32],
                               "pbd": "aia_lev1_211a_2011_02_15t01_46_00_62z_image_lev1_fits.fits"},
             "corpita_fig8e": {"tr": hek.attrs.Time('2011-02-16 14:22:36', '2011-02-16 14:39:48'),
                               "accum": 3,
                               "result": 0,
                               "lon_index": 5,
                               "listindex": [10, 12]}}

sunday_name = {"previous1": "better name",
               "corpita_fig4": "better name"}