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

        "cadence": 12.,  # seconds

        "start_time_offset": 0.0,  # seconds

        "hglt_obs": 0. * u.degree,  # degrees
        "rotation": 360. / (27. * 86400.) * u.degree / u.s,  # degrees/s, rigid solar rotation

        # Wave parameters that are initial conditions
        "direction": -205. * u.degree,  # degrees, measured CCW from HG +latitude
        "epi_lat": 0 * u.degree,  # degrees, HG latitude of wave epicenter
        "epi_lon": 0 * u.degree,  # degrees, HG longitude of wave epicenter

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
        "speed": np.asarray([9.33e5, 0.0, 0.0]) * m2deg * u.m / u.s,
        "acceleration": 0.0e3 * m2deg * u.m / u.s / u.s,

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

        "clean_nans": False,

        # HG grid, probably would only want to change the bin sizes
        "lat_min": -90. * u.degree,
        "lat_max": 90. * u.degree,
        "lat_bin": 0.2 * u.degree,
        "lon_min": 0.0 * u.degree,
        "lon_max": 360. * u.degree,
        "lon_bin": 1.0 * u.degree,

        # HPC grid, probably would only want to change the bin sizes
        "hpcx_min": -1228.8 * u.arcsec,
        "hpcx_max": 1228.8 * u.arcsec,
        "hpcx_bin": 2.4 * u.arcsec,
        "hpcy_min": -1228.8 * u.arcsec,
        "hpcy_max": 1228.8 * u.arcsec,
        "hpcy_bin": 2.4 * u.arcsec,
        "xnum": 800*u.pix,
        "ynum": 800*u.pix
    }

    # The wave normalization is set to a 1.0 - a low SNR wave.
    lowsnr = copy.deepcopy(basic_wave)
    lowsnr["wave_normalization"] = 1.0
    lowsnr["name"] = "lowsnr"

    # A version of lowsnr with the following changes:
    # (a) full 360 degrees wave
    lowsnr_full360 = copy.deepcopy(lowsnr)
    lowsnr_full360["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    lowsnr_full360["name"] = "lowsnr_full360"

    # A version of lowsnr_full360 with the following changes:
    # (a) half the speed of lowsnr_full360
    lowsnr_full360_slow = copy.deepcopy(lowsnr_full360)
    lowsnr_full360_slow["speed"] = lowsnr_full360_slow["speed"] / 2.0
    lowsnr_full360_slow["name"] = "lowsnr_full360_slow"

    # A version of lowsnr_full360_slow with the following changes
    # (a) no solar rotation
    lowsnr_full360_slow_nosolarrotation = copy.deepcopy(lowsnr_full360_slow)
    lowsnr_full360_slow_nosolarrotation["rotation"] = 0.0 * u.degree / u.s
    lowsnr_full360_slow_nosolarrotation["name"] = "lowsnr_full360_slow_nosolarrotation"

    # A version of lowsnr_full360_slow with the following changes
    # (a) displaced center
    lowsnr_full360_slow_displacedcenter = copy.deepcopy(lowsnr_full360_slow)
    lowsnr_full360_slow_displacedcenter['epi_lat'] = 45 * u.degree
    lowsnr_full360_slow_displacedcenter['epi_lon'] = 54 * u.degree
    lowsnr_full360_slow_displacedcenter['name'] = 'lowsnr_full360_slow_displacedcenter'

    # A version of lowsnr_full360_slow_nosolarrotation with the following
    # changes
    # (a) displaced center
    lowsnr_full360_slow_nosolarrotation_displacedcenter = copy.deepcopy(lowsnr_full360_slow_nosolarrotation)
    lowsnr_full360_slow_nosolarrotation_displacedcenter['epi_lat'] = 45 * u.degree
    lowsnr_full360_slow_nosolarrotation_displacedcenter['epi_lon'] = 54 * u.degree
    lowsnr_full360_slow_nosolarrotation_displacedcenter['name'] = 'lowsnr_full360_slow_nosolarrotation_displacedcenter'

    # A version of lowsnr_full360_slow with the following changes
    # (a) acceleration
    lowsnr_full360_slow_accelerated = copy.deepcopy(lowsnr_full360_slow)
    lowsnr_full360_slow_accelerated['acceleration'] = 1.5e3 * m2deg * u.m / u.s / u.s
    lowsnr_full360_slow_accelerated['name'] = 'lowsnr_full360_slow_accelerated'

    # A version of lowsnr_full360_slow_accelerated with the following changes:
    # (a) displaced center
    lowsnr_full360_slow_accelerated_displacedcenter = copy.deepcopy(lowsnr_full360_slow_accelerated)
    lowsnr_full360_slow_accelerated_displacedcenter['epi_lat'] = 45 * u.degree
    lowsnr_full360_slow_accelerated_displacedcenter['epi_lon'] = 54 * u.degree
    lowsnr_full360_slow_accelerated_displacedcenter['name'] = 'lowsnr_full360_slow_accelerated_displacedcenter'

    return {'basic_wave': basic_wave,
            "lowsnr": lowsnr,
            "lowsnr_full360": lowsnr_full360,
            "lowsnr_full360_slow": lowsnr_full360_slow,
            "lowsnr_full360_slow_nosolarrotation": lowsnr_full360_slow_nosolarrotation,
            "lowsnr_full360_slow_displacedcenter": lowsnr_full360_slow_displacedcenter,
            "lowsnr_full360_slow_nosolarrotation_displacedcenter": lowsnr_full360_slow_nosolarrotation_displacedcenter,
            "lowsnr_full360_slow_accelerated": lowsnr_full360_slow_accelerated,
            "lowsnr_full360_slow_accelerated_displacedcenter": lowsnr_full360_slow_accelerated_displacedcenter}

