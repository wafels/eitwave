#
# File that contains simulated wave profiles.
#
import copy
import numpy as np
import astropy.units as u
import aware_constants

m2deg = aware_constants.m2deg


def waves():

    #
    # A simple wave
    #
    acceleration = 1.5e3 * m2deg * u.m / u.s / u.s
    hisnr_wave_normalization = 1000.0

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
        "width": np.asarray([90., 0.0, 0.0]) * u.degree,  # degrees, full angle in azimuth, centered at 'direction'
        "wave_thickness": np.asarray([6.0e6, 0.0, 0.0]) * m2deg * u.m,  # degrees, sigma of Gaussian profile in longitudinal direction
        "wave_normalization": 10.,  # integrated value of the 1D Gaussian profile
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
        "lon_min": -180 * u.degree,
        "lon_max": 180. * u.degree,
        "lon_bin": 1.0 * u.degree,

        # HPC grid, probably would only want to change the bin sizes
        "hpcx_min": -1228.8 * u.arcsec,
        "hpcx_max": 1228.8 * u.arcsec,
        "hpcx_bin": 2.4 * u.arcsec,
        "hpcy_min": -1228.8 * u.arcsec,
        "hpcy_max": 1228.8 * u.arcsec,
        "hpcy_bin": 2.4 * u.arcsec,
        "xnum": 1024*u.pix,
        "ynum": 1024*u.pix
    }

    # The wave normalization is set to the same level as the background.
    wnbacksnr = copy.deepcopy(basic_wave)
    wnbacksnr["wave_normalization"] = 10.0
    wnbacksnr["noise_mean"] = wnbacksnr["wave_normalization"]
    wnbacksnr["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wnbacksnr["speed"] = basic_wave["speed"] / 2.0
    wnbacksnr["acceleration"] = acceleration
    wnbacksnr["rotation"] = 0.0 * u.degree / u.s
    wnbacksnr["name"] = "wnbacksnr"

    # Same as wnbacksnr, except with a displaced center.
    wnbacksnr_displacedcenter = copy.deepcopy(wnbacksnr)
    wnbacksnr_displacedcenter['epi_lat'] = 22 * u.degree
    wnbacksnr_displacedcenter['epi_lon'] = 33 * u.degree
    wnbacksnr_displacedcenter["name"] = "wnbacksnr_displacedcenter"

    # Super low: the wave normalization is set to the same level as the background.
    superlowsnr = copy.deepcopy(basic_wave)
    superlowsnr["wave_normalization"] = 5.0
    superlowsnr["noise_mean"] = superlowsnr["wave_normalization"]
    superlowsnr["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    superlowsnr["speed"] = basic_wave["speed"] / 2.0
    superlowsnr["acceleration"] = acceleration
    superlowsnr["rotation"] = 0.0 * u.degree / u.s
    superlowsnr["name"] = "superlowsnr"

    # Same as wnbacksnr, except with a displaced center.
    superlowsnr_displacedcenter = copy.deepcopy(superlowsnr)
    superlowsnr_displacedcenter['epi_lat'] = 22 * u.degree
    superlowsnr_displacedcenter['epi_lon'] = 33 * u.degree
    superlowsnr_displacedcenter["name"] = "superlowsnr_displacedcenter"

    # The wave normalization is set to a 2.5 - a low SNR wave.
    lowsnr = copy.deepcopy(basic_wave)
    lowsnr["wave_normalization"] = 2.5
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
    # (a) acceleration
    lowsnr_full360_slow_nosolarrotation_accelerated = copy.deepcopy(lowsnr_full360_slow_nosolarrotation)
    lowsnr_full360_slow_nosolarrotation_accelerated["acceleration"] = acceleration
    lowsnr_full360_slow_nosolarrotation_accelerated["name"] = "lowsnr_full360_slow_nosolarrotation_accelerated"

    # A version of lowsnr_full360_slow_nosolarrotation_accelerated with the following changes
    # (a) displaced center
    lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter = copy.deepcopy(lowsnr_full360_slow_nosolarrotation_accelerated)
    lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter['epi_lat'] = 22 * u.degree
    lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter['epi_lon'] = 33 * u.degree
    lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter['name'] = 'lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter'

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
    lowsnr_full360_slow_accelerated['acceleration'] = acceleration
    lowsnr_full360_slow_accelerated['name'] = 'lowsnr_full360_slow_accelerated'

    # A version of lowsnr_full360_slow_accelerated with the following changes:
    # (a) displaced center
    lowsnr_full360_slow_accelerated_displacedcenter = copy.deepcopy(lowsnr_full360_slow_accelerated)
    lowsnr_full360_slow_accelerated_displacedcenter['epi_lat'] = 45 * u.degree
    lowsnr_full360_slow_accelerated_displacedcenter['epi_lon'] = 54 * u.degree
    lowsnr_full360_slow_accelerated_displacedcenter['name'] = 'lowsnr_full360_slow_accelerated_displacedcenter'

    # A version of lowsnr_full360_slow with the following changes
    # (a) displaced center
    basicwave_full360_slow_displacedcenter = copy.deepcopy(basic_wave)
    basicwave_full360_slow_displacedcenter["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    basicwave_full360_slow_displacedcenter['speed'] = basicwave_full360_slow_displacedcenter['speed'] / 2.0
    basicwave_full360_slow_displacedcenter['epi_lat'] = 45 * u.degree
    basicwave_full360_slow_displacedcenter['epi_lon'] = 54 * u.degree
    basicwave_full360_slow_displacedcenter['name'] = 'basicwave_full360_slow_displacedcenter'

    # High signal to noise ratio
    # A version of lowsnr_full360 with the following changes:
    # (a) full 360 degrees wave
    hisnr_full360_slow = copy.deepcopy(lowsnr_full360_slow)
    hisnr_full360_slow["wave_normalization"] = hisnr_wave_normalization
    hisnr_full360_slow["name"] = "hisnr_full360_slow"

    # A version of hisnr_full360 with the following changes:
    # (a) no solar rotation
    hisnr_full360_nosolarrotation_slow = copy.deepcopy(hisnr_full360_slow)
    hisnr_full360_nosolarrotation_slow["noise_type"] = None
    hisnr_full360_nosolarrotation_slow["rotation"] = 0.0 * u.degree / u.s
    hisnr_full360_nosolarrotation_slow["name"] = "hisnr_full360_nosolarrotation_slow"

    # A version of hisnr_full360_nosolarrotation with the following changes:
    # (a) acceleration
    hisnr_full360_slow_nosolarrotation_accelerated = copy.deepcopy(hisnr_full360_nosolarrotation_slow)
    hisnr_full360_slow_nosolarrotation_accelerated['acceleration'] = acceleration
    hisnr_full360_slow_nosolarrotation_accelerated["name"] = "hisnr_full360_slow_nosolarrotation_accelerated"

    # A version of hisnr_full360_nosolarrotation with the following changes:
    # (a) acceleration
    hisnr_full360_nosolarrotation_acceleration_slow2 = copy.deepcopy(hisnr_full360_slow_nosolarrotation_accelerated)
    hisnr_full360_nosolarrotation_acceleration_slow2['speed'] = lowsnr_full360_slow["speed"] / 4.0
    hisnr_full360_nosolarrotation_acceleration_slow2["name"] = "hisnr_full360_nosolarrotation_acceleration_slow2"
    # (a) acceleration
    hisnr_full360_nosolarrotation_acceleration_slow3 = copy.deepcopy(hisnr_full360_nosolarrotation_acceleration_slow2)
    hisnr_full360_nosolarrotation_acceleration_slow3['speed'] = lowsnr_full360_slow["speed"] / 8.0
    hisnr_full360_nosolarrotation_acceleration_slow3["name"] = "hisnr_full360_nosolarrotation_acceleration_slow3"

    # A version of lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter with the following changes:
    # (a) high signal to noise
    hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter = copy.deepcopy(lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter)
    hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter['name'] = 'hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter'
    hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter["wave_normalization"] = hisnr_wave_normalization

    return {'basic_wave': basic_wave,
            "lowsnr": lowsnr,
            "lowsnr_full360": lowsnr_full360,
            "lowsnr_full360_slow": lowsnr_full360_slow,
            "lowsnr_full360_slow_nosolarrotation": lowsnr_full360_slow_nosolarrotation,
            "lowsnr_full360_slow_displacedcenter": lowsnr_full360_slow_displacedcenter,
            "lowsnr_full360_slow_nosolarrotation_displacedcenter": lowsnr_full360_slow_nosolarrotation_displacedcenter,
            "lowsnr_full360_slow_accelerated": lowsnr_full360_slow_accelerated,
            "lowsnr_full360_slow_nosolarrotation_accelerated": lowsnr_full360_slow_nosolarrotation_accelerated,
            "lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter": lowsnr_full360_slow_nosolarrotation_accelerated_displacedcenter,
            "lowsnr_full360_slow_accelerated_displacedcenter": lowsnr_full360_slow_accelerated_displacedcenter,
            "basicwave_full360_slow_displacedcenter": basicwave_full360_slow_displacedcenter,
            "hisnr_full360_slow": hisnr_full360_slow,
            "hisnr_full360_nosolarrotation_slow": hisnr_full360_nosolarrotation_slow,
            "hisnr_full360_slow_nosolarrotation_accelerated": hisnr_full360_slow_nosolarrotation_accelerated,
            "hisnr_full360_nosolarrotation_acceleration_slow2": hisnr_full360_nosolarrotation_acceleration_slow2,
            "hisnr_full360_nosolarrotation_acceleration_slow3": hisnr_full360_nosolarrotation_acceleration_slow3,
            "hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter": hisnr_full360_slow_nosolarrotation_accelerated_displacedcenter,
            "wnbacksnr": wnbacksnr,
            "wnbacksnr_displacedcenter": wnbacksnr_displacedcenter,
            "superlowsnr": superlowsnr,
            "superlowsnr_displacedcenter": superlowsnr_displacedcenter}

