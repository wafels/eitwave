#
# File that contains simulated wave profiles.
#
import copy
import numpy as np
import astropy.units as u
import aware_utils

m2deg = aware_utils.m2deg


def waves(lon_start=-180.0 * u.degree):

    #
    # A simple wave
    #
    basic_wave = {
        "name": 'basic wave',

        "cadence": 12.,  # seconds

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
        "lon_min": lon_start,
        "lon_max": lon_start + 360. * u.degree,
        "lon_bin": 1.0 * u.degree,

        # HPC grid, probably would only want to change the bin sizes
        "hpcx_min": -1228.8 * u.arcsec,
        "hpcx_max": 1228.8 * u.arcsec,
        "hpcx_bin": 2.4 * u.arcsec,
        "hpcy_min": -1228.8 * u.arcsec,
        "hpcy_max": 1228.8 * u.arcsec,
        "hpcy_bin": 2.4 * u.arcsec
    }

    # Remove the effect of solar rotation
    no_solar_rotation = copy.deepcopy(basic_wave)
    no_solar_rotation["rotation"] = 0.0 * u.degree / u.s
    no_solar_rotation["name"] = 'no solar rotation'

    # No noise - a wave on a blank Sun
    no_noise = copy.deepcopy(basic_wave)
    no_noise["noise_scale"] = 0.0
    no_noise["name"] = "no noise"

    # No noise, no solar rotation - a wave on a blank Sun
    no_noise_no_solar_rotation = copy.deepcopy(basic_wave)
    no_noise_no_solar_rotation["rotation"] = 0.0 * u.degree / u.s
    no_noise_no_solar_rotation["name"] = 'no noise, no solar rotation'
    no_noise_no_solar_rotation["noise_scale"] = 0.0

    # No noise, no solar rotation - a wave on a blank Sun
    no_noise_no_solar_rotation_slow = copy.deepcopy(basic_wave)
    no_noise_no_solar_rotation_slow["rotation"] = 0.0 * u.degree / u.s
    no_noise_no_solar_rotation_slow["name"] = 'no noise, no solar rotation'
    no_noise_no_solar_rotation_slow["noise_scale"] = 0.0
    no_noise_no_solar_rotation_slow["speed"] = np.asarray([9.33e5, 0.0, 0.0]) * m2deg * u.m / u.s / 2.0

    # Low noise - a wave on a low noise Sun
    low_noise = copy.deepcopy(basic_wave)
    low_noise["noise_scale"] = 0.0001
    low_noise["name"] = "low noise"

    # Low noise full - a wave on a low noise Sun
    low_noise_full = copy.deepcopy(low_noise)
    low_noise_full["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    low_noise_full["name"] = "low noise full"

    # The wave normalization is set to a 1.0 - a low SNR wave.
    wavenorm1 = copy.deepcopy(basic_wave)
    wavenorm1["wave_normalization"] = 1.0
    wavenorm1["name"] = "wavenorm1"

    # A version of wavenorm1 with the following changes:
    # (a) full 360 degree wave
    # (b) no solar rotation.
    wavenorm2 = copy.deepcopy(wavenorm1)
    wavenorm2["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wavenorm2["rotation"] = 0.0 * u.degree / u.s
    wavenorm2["name"] = "wavenorm2 (no solar rotation)"

    # A version of wavenorm2 with the following changes:
    # (a) The wave is twice as thick as wavenorwm2.
    wavenorm3 = copy.deepcopy(wavenorm2)
    wavenorm3["wave_thickness"] = np.asarray([1.2e7, 0.0, 0.0]) * m2deg * u.degree
    wavenorm3["name"] = "wavenorm3 (twice as thick)"

    # A version of wavenorm1 with the following changes:
    # (a) full 360 degrees wave
    wavenorm4 = copy.deepcopy(wavenorm1)
    wavenorm4["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wavenorm4["name"] = "wavenorm4 (with solar rotation)"

    # A version of wavenorm4 with the following changes:
    # (a) half the speed of wavenorm4
    # (b) No solar rotation
    wavenorm4_slow = copy.deepcopy(wavenorm4)
    wavenorm4_slow["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wavenorm4_slow["rotation"] = 0.0 * u.degree / u.s
    wavenorm4_slow["name"] = "wavenorm4_slow (no solar rotation)"
    wavenorm4_slow["speed"] = wavenorm4_slow["speed"] / 2.0

    # A version of wavenorm4_slow with the following changes:
    # (a) half the speed of wavenorm4
    wavenorm4_slow_wsr = copy.deepcopy(wavenorm4)
    wavenorm4_slow_wsr["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wavenorm4_slow_wsr["name"] = "wavenorm4_slow"
    wavenorm4_slow_wsr["speed"] = wavenorm4_slow["speed"] / 2.0

    # A version of wavenorm4_slow_wsr with the following changes:
    # (a) off center wave
    wavenorm4_slow_wsr_offcenter = copy.deepcopy(wavenorm4_slow_wsr)
    wavenorm4_slow_wsr_offcenter["epi_lat"] = 53.8 * u.degree  # degrees, HG latitude of wave epicenter
    wavenorm4_slow_wsr_offcenter["epi_lon"] = 61.1 * u.degree  # degrees, HG latitude of wave epicenter

    # A version of wavenorm4 with the following changes:
    # (a) quarter the speed of wavenorm4
    wavenorm4_vslow = copy.deepcopy(wavenorm4)
    wavenorm4_vslow["width"] = np.asarray([360., 0.0, 0.0]) * u.degree
    wavenorm4_vslow["name"] = "wavenorm4_vslow (with solar rotation)"
    wavenorm4_vslow["speed"] = wavenorm4_vslow["speed"] / 4.0

    # A version of wavenorm4_slow with the following changes:
    # (a) displaced center
    wavenorm4_slow_displaced = copy.deepcopy(wavenorm4_slow)
    wavenorm4_slow_displaced['epi_lat'] = 45 * u.degree
    wavenorm4_slow_displaced['epi_lon'] = 45 * u.degree

    return {'basic_wave': basic_wave,
            'no_solar_rotation': no_solar_rotation,
            "no_noise": no_noise,
            "no_noise_no_solar_rotation": no_noise_no_solar_rotation,
            "no_noise_no_solar_rotation_slow": no_noise_no_solar_rotation_slow,
            "low_noise": low_noise,
            "low_noise_full": low_noise_full,
            "wavenorm1": wavenorm1,
            "wavenorm2": wavenorm2,
            "wavenorm3": wavenorm3,
            "wavenorm4": wavenorm4,
            "wavenorm4_slow": wavenorm4_slow,
            "wavenorm4_slow_wsr": wavenorm4_slow_wsr,
            "wavenorm4_slow_wsr_offcenter": wavenorm4_slow_wsr_offcenter,
            "wavenorm4_vslow": wavenorm4_vslow,
            "wavenorm4_slow_displaced": wavenorm4_slow_displaced}