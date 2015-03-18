#
# File that contains simulated wave profiles.
#
import copy
import numpy as np
import astropy.units as u
import aware_utils

m2deg = aware_utils.m2deg

def waves():

#
# A simple wave
#
    basic_wave = {
        "name": 'basic wave',

        "cadence": 12., #seconds

        "hglt_obs": 0. * u.degree, #degrees
        "rotation": 360. / (27. * 86400.) * u.degree / u.s, #degrees/s, rigid solar rotation

        # Wave parameters that are initial conditions
        "direction": -205. * u.degree, #degrees, measured CCW from HG +latitude
        "epi_lat": 0 * u.degree, #degrees, HG latitude of wave epicenter
        "epi_lon": 0 * u.degree, #degrees, HG longitude of wave epicenter

        # The following quantities are wave parameters that can evolve over time.  The
        # evolution rates are allowed to have constant with time, linear with time,
        # and quadratic with time variations.  The coefficients which govern this
        # temporal behaviour are quoted in arrays with a maximum of three elements
        # The first element is constant in time (quantity).
        # The second element (if present) is linear in time (quantity/second).
        # The third element (if present) is quadratic in time (quantity/second/second).
        # Be very careful of non-physical behavior.
        "width": np.asarray([90., 0.0, 0.0]) * u.degree, #degrees, full angle in azimuth, centered at 'direction'
        "wave_thickness": np.asarray([6.0e6, 0.0, 0.0]) * m2deg * u.degree, #degrees, sigma of Gaussian profile in longitudinal direction
        "wave_normalization": [10.], #integrated value of the 1D Gaussian profile
        # sim_speed #degrees/s, make sure that wave propagates all the way to lat_min for polynomial speed
        "speed": np.asarray([9.33e5, 0.0, 0.0]) * m2deg * u.degree / u.s,

        # Random noise parameters
        "noise_type": "Poisson", #can be None, "Normal", or "Poisson"
        "noise_scale": 1.,
        "noise_mean": 1.,
        "noise_sdev": 1.,

        # Structured noise parameters
        "struct_type": "None", #can be None, "Arcs", or "Random"
        "struct_scale": 5.,
        "struct_num": 10,
        "struct_seed": 13092,

        "clean_nans": True,

        # HG grid, probably would only want to change the bin sizes
        "lat_min": -90. * u.degree,
        "lat_max": 90. * u.degree,
        "lat_bin": 0.2 * u.degree,
        "lon_min": -180. * u.degree,
        "lon_max": 180. * u.degree,
        "lon_bin": 1.0 * u.degree,

        # HPC grid, probably would only want to change the bin sizes
        "hpcx_min": -1228.8 * u.arcsec,
        "hpcx_max": 1228.8 * u.arcsec,
        "hpcx_bin": 2.4 * u.arcsec,
        "hpcy_min": -1228.8 * u.arcsec,
        "hpcy_max": 1228.8 * u.arcsec,
        "hpcy_bin": 2.4 * u.arcsec
    }

    no_solar_rotation = copy.deepcopy(basic_wave)
    no_solar_rotation["rotation"] = 0.0 * u.degree / u.s
    no_solar_rotation["name"] = 'no solar rotation'

    no_noise = copy.deepcopy(basic_wave)
    no_noise["noise_scale"] = 0.0
    no_noise["name"] = "no noise"

    low_noise = copy.deepcopy(basic_wave)
    low_noise["noise_scale"] = 0.0001
    low_noise["name"] = "low noise"

    return {'basic_wave': basic_wave,
            'no_solar_rotation': no_solar_rotation,
            "no_noise": no_noise,
            "low_noise": low_noise}