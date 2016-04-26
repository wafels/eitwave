import os
import numpy as np
import astropy.units as u
import sunpy.sun as sun


#
# Constants used in other parts of aware
#
solar_circumference_per_degree = 2 * np.pi * sun.constants.radius.to('m') / (360.0 * u.degree)
m2deg = 1.0 / solar_circumference_per_degree
solar_circumference_per_degree_in_km = solar_circumference_per_degree.to('km/deg') * u.degree

score_long_velocity_range = [1.0, 2000.0] * u.km/u.s
score_long_acceleration_range = [-2.0, 2.0] * u.km/u.s/u.s

eitwave_data_root = os.path.expanduser('~/Data/eitwave')
