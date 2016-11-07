"""
Simulates an EUV wave
"""

from __future__ import absolute_import
from copy import deepcopy
import datetime
import numpy as np
from scipy.special import ndtr
from scipy.interpolate import griddata
import matplotlib.cm as cm
from astropy.visualization import LinearStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.units as u
from sunpy import wcs
from sunpy.map import Map, MapMeta
import sunpy.sun.sun as sun
from sunpy.time import parse_time
from map_hpc_hg_transforms import euler_zyz, map_hg_to_hpc_rotate


__all__ = ["simulate", "simulate_raw", "transform", "add_noise"]

__authors__ = ["Albert Shih"]
__email__ = "albert.y.shih@nasa.gov"

#
# Has value 0.5 in original wave2d.py code
# This makes lon_min the left edge of the first bin
# This makes lat_min the left edge of the first bin
#
# Has value 1.0 in util.py
#
#
crpix12_value_for_HG = 0.5

#
# Has value 0.5 in original wave2d.py code
# This makes hpcx_min the left edge of the first bin
#
# Has value 1.0 in util.py
# This makes hpcx.min() the center of the first bin.
# This makes hpct.min() the center of the first bin.
#
crpix12_value_for_HPC = 0.5


# Initial date for the simulated data
BASE_DATE = parse_time('2020-01-01T00:00:00.00')
BASE_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def prep_coeff(coeff, order=2):
    """
    Prepares polynomial coefficients out to a certain order, outputs as ndarray
    """
    new_coeff = np.zeros(order+1)
    if type(coeff) == list or type(coeff) == np.ndarray or type(coeff) == u.quantity.Quantity:
        size = min(len(coeff), len(new_coeff))
        new_coeff[0:size] = coeff[0:size]
    else:
        new_coeff[0] = coeff
    return new_coeff


def prep_speed_coeff(speed, acceleration, jerk=0.0):
    """
    Prepares speed, acceleration, jerk
    """
    return np.asarray([speed[0].value, acceleration.value, jerk])


def simulate_raw(params, steps, verbose=False):
    """
    Simulate data in HG' coordinates
    
    HG' = HG, except center at wave epicenter
    """
    cadence = params["cadence"]
    direction = 180. + params["direction"].to('degree').value
    
    width_coeff = prep_coeff(params["width"])
    wave_thickness_coeff = prep_coeff(params["wave_thickness"])
    wave_normalization_coeff = prep_coeff(params["wave_normalization"])
    speed_coeff = prep_speed_coeff(params["speed"], params["acceleration"])

    lat_min = params["lat_min"].to('degree').value
    lat_max = params["lat_max"].to('degree').value
    lat_bin = params["lat_bin"].to('degree').value
    lon_min = params["lon_min"].to('degree').value
    lon_max = params["lon_max"].to('degree').value
    lon_bin = params["lon_bin"].to('degree').value

    # This roundabout approach recalculates lat_bin and lon_bin to produce
    # equally sized bins to exactly span the min/max ranges
    lat_num = int(round((lat_max-lat_min)/lat_bin))
    lat_edges, lat_bin = np.linspace(lat_min, lat_max, lat_num+1, retstep=True)

    lon_num = int(round((lon_max-lon_min)/lon_bin))
    lon_edges, lon_bin = np.linspace(lon_min, lon_max, lon_num+1, retstep=True)

    # Propagates from 90. down to lat_min, irrespective of lat_max
    p = np.poly1d([speed_coeff[2]/3., speed_coeff[1]/2., speed_coeff[0],
                   -(90.-lat_min)])
    # p = np.poly1d([0.0, speed_coeff[1], speed_coeff[2]/2.,
    #               -(90.-lat_min)])
    # Will fail if wave does not propagate all the way to lat_min
    # duration = p.r[np.logical_and(p.r.real > 0, p.r.imag == 0)][0]
    
    # steps = int(duration/cadence)+1
    # if steps > params["max_steps"]:
    #    steps = params["max_steps"]
    
    # Maybe used np.poly1d() instead to do the polynomial calculation?
    time = params["start_time_offset"] + np.arange(steps)*cadence
    time_powers = np.vstack((time**0, time**1, time**2))
    
    width = np.dot(width_coeff, time_powers).ravel()
    wave_thickness = np.dot(wave_thickness_coeff, time_powers).ravel()
    wave_normalization = np.dot(wave_normalization_coeff, time_powers).ravel()

    #Position
    #Propagates from 90., irrespective of lat_max
    wave_peak = 90.-(p(time)+(90.-lat_min))

    out_of_bounds = np.logical_or(wave_peak < lat_min, wave_peak > lat_max)
    if out_of_bounds.any():
        steps = np.where(out_of_bounds)[0][0]

    # Storage for the wave maps
    wave_maps = []

    # Header of the wave maps
    dict_header = {
        "CDELT1": lon_bin,
        "NAXIS1": lon_num,
        "CRVAL1": lon_min,
        "CRPIX1": crpix12_value_for_HG,
        "CUNIT1": "deg",
        "CTYPE1": "HG",
        "CDELT2": lat_bin,
        "NAXIS2": lat_num,
        "CRVAL2": lat_min,
        "CRPIX2": crpix12_value_for_HG,
        "CUNIT2": "deg",
        "CTYPE2": "HG",
        "HGLT_OBS": 0.0,  # (sun.heliographic_solar_center(BASE_DATE))[1],  # the value of HGLT_OBS from Earth at the given date
        "CRLN_OBS": 0.0,  # (sun.heliographic_solar_center(BASE_DATE))[0],  # the value of CRLN_OBS from Earth at the given date
        "DSUN_OBS": sun.sunearth_distance(BASE_DATE.strftime(BASE_DATE_FORMAT)).to('m').value,
        "DATE_OBS": BASE_DATE.strftime(BASE_DATE_FORMAT),
        "EXPTIME": 1.0
    }

    if verbose:
        print("  * Simulating "+str(steps)+" raw maps.")

    for istep in range(0, steps):

        # Current datetime
        current_datetime = BASE_DATE + datetime.timedelta(seconds=time[istep])

        # Update the header to set the correct observation time and earth-sun
        # distance
        dict_header['DATE_OBS'] = current_datetime.strftime(BASE_DATE_FORMAT)

        # Update the Earth-Sun distance
        dict_header['DSUN_OBS'] = sun.sunearth_distance(dict_header['DATE_OBS']).to('m').value

        # Update the heliographic latitude
        dict_header['HGLT_OBS'] = 0.0  # (sun.heliographic_solar_center(dict_header['DATE_OBS']))[1].to('degree').value

        # Update the heliographic longitude
        dict_header['CRLN_OBS'] = 0.0  # (sun.heliographic_solar_center(dict_header['DATE_OBS']))[0].to('degree').value

        # Gaussian profile in longitudinal direction
        # Does not take into account spherical geometry (i.e., change in area
        # element)
        if wave_thickness[istep] <= 0:
            print("  * ERROR: wave thickness is non-physical!")
        z = (lat_edges-wave_peak[istep])/wave_thickness[istep]
        wave_1d = wave_normalization[istep]*(ndtr(np.roll(z, -1))-ndtr(z))[0:lat_num]
        wave_1d /= lat_bin
        
        wave_lon_min = direction-width[istep]/2
        wave_lon_max = direction+width[istep]/2

        if width[istep] < 360.:
            # Do these need to be np.remainder() instead?
            wave_lon_min_mod = ((wave_lon_min+180.) % 360.)-180.
            wave_lon_max_mod = ((wave_lon_max+180.) % 360.)-180.
            
            index1 = np.arange(lon_num+1)[np.roll(lon_edges, -1) > min(wave_lon_min_mod, wave_lon_max_mod)][0]
            index2 = np.roll(np.arange(lon_num+1)[lon_edges < max(wave_lon_min_mod, wave_lon_max_mod)], 1)[0]
    
            wave_lon = np.zeros(lon_num)
            wave_lon[index1+1:index2] = 1.
            # Possible weirdness if index1 == index2
            wave_lon[index1] += (lon_edges[index1+1]-min(wave_lon_min_mod, wave_lon_max_mod))/lon_bin
            wave_lon[index2] += (max(wave_lon_min_mod, wave_lon_max_mod)-lon_edges[index2])/lon_bin
            
            if wave_lon_min_mod > wave_lon_max_mod:
                wave_lon = 1.-wave_lon
        else:
            wave_lon = np.ones(lon_num)
        
        # Could be accomplished with np.dot() without casting as matrices?
        wave = np.mat(wave_1d).T*np.mat(wave_lon)

        # Create the new map
        new_map = Map(wave, MapMeta(dict_header))
        new_map.plot_settings = {'cmap': cm.gray,
                                 'norm': ImageNormalize(stretch=LinearStretch()),
                                 'interpolation': 'nearest',
                                 'origin': 'lower'
                                 }
        # Update the list of maps
        wave_maps += [new_map]

    return Map(wave_maps, cube=True)


def transform(params, wave_maps, verbose=False):
    """
    Transform raw data in HG' coordinates to HPC coordinates
    
    HG' = HG, except center at wave epicenter
    """
    solar_rotation_rate = params["rotation"]

    hglt_obs = params["hglt_obs"].to('degree').value
    # crln_obs = params["crln_obs"]
    
    epi_lat = params["epi_lat"].to('degree').value
    epi_lon = params["epi_lon"].to('degree').value

    # Parameters for the HPC co-ordinates
    hpcx_min = params["hpcx_min"].to('arcsec').value
    hpcx_max = params["hpcx_max"].to('arcsec').value
    hpcx_bin = params["hpcx_bin"].to('arcsec').value

    hpcy_min = params["hpcy_min"].to('arcsec').value
    hpcy_max = params["hpcy_max"].to('arcsec').value
    hpcy_bin = params["hpcy_bin"].to('arcsec').value

    hpcx_num = int(round((hpcx_max-hpcx_min)/hpcx_bin))
    hpcy_num = int(round((hpcy_max-hpcy_min)/hpcy_bin))

    # Storage for the HPC version of the input maps
    wave_maps_transformed = []

    # The properties of this map are used in the transform
    smap = wave_maps[0]

    # Basic dictionary version of the HPC map header
    dict_header = {
        "CDELT1": hpcx_bin,
        "NAXIS1": hpcx_num,
        "CRVAL1": hpcx_min,
        "CRPIX1": crpix12_value_for_HPC,
        "CUNIT1": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CDELT2": hpcy_bin,
        "NAXIS2": hpcy_num,
        "CRVAL2": hpcy_min,
        "CRPIX2": crpix12_value_for_HPC,
        "CUNIT2": "arcsec",
        "CTYPE2": "HPLT-TAN",
        "HGLT_OBS": hglt_obs,
        "CRLN_OBS": smap.carrington_longitude.to('degree').value,
        "DSUN_OBS": sun.sunearth_distance(BASE_DATE.strftime(BASE_DATE_FORMAT)).to('meter').value,
        "DATE_OBS": BASE_DATE.strftime(BASE_DATE_FORMAT),
        "EXPTIME": 1.0
    }
    start_date = smap.date

    # Origin grid, HG'
    lon_grid, lat_grid = wcs.convert_pixel_to_data([smap.data.shape[1], smap.data.shape[0]],
                                                   [smap.scale.x.value, smap.scale.y.value],
                                                   [smap.reference_pixel.x.value, smap.reference_pixel.y.value],
                                                   [smap.reference_coordinate.x.value, smap.reference_coordinate.y.value])

    # Origin grid, HG' to HCC'
    # HCC' = HCC, except centered at wave epicenter
    x, y, z = wcs.convert_hg_hcc(lon_grid, lat_grid,
                                 b0_deg=smap.heliographic_latitude.to('degree').value,
                                 l0_deg=smap.carrington_longitude.to('degree').value,
                                 z=True)

    # Origin grid, HCC' to HCC''
    # Moves the wave epicenter to initial conditions
    # HCC'' = HCC, except assuming that HGLT_OBS = 0
    zxy_p = euler_zyz((z, x, y),
                      (epi_lon, 90.-epi_lat, 0.))

    # Destination HPC grid
    hpcx_grid, hpcy_grid = wcs.convert_pixel_to_data([dict_header['NAXIS1'], dict_header['NAXIS2']],
                                                     [dict_header['CDELT1'], dict_header['CDELT2']],
                                                     [dict_header['CRPIX1'], dict_header['CRPIX2']],
                                                     [dict_header['CRVAL1'], dict_header['CRVAL2']])

    for icwm, current_wave_map in enumerate(wave_maps):
        print(icwm, len(wave_maps))
        # Elapsed time
        td = parse_time(current_wave_map.date) - parse_time(start_date)

        # Update the header
        dict_header['DATE_OBS'] = current_wave_map.date
        dict_header['DSUN_OBS'] = current_wave_map.dsun.to('m').value

        # Origin grid, HCC'' to HCC
        # Moves the observer to HGLT_OBS and adds rigid solar rotation
        total_seconds = u.s * (td.microseconds + (td.seconds + td.days * 24.0 * 3600.0) * 10.0**6) / 10.0**6
        solar_rotation = (total_seconds * solar_rotation_rate).to('degree').value
        zpp, xpp, ypp = euler_zyz(zxy_p,
                                  (0., hglt_obs, solar_rotation))

        # Origin grid, HCC to HPC (arcsec)
        xx, yy = wcs.convert_hcc_hpc(xpp, ypp,
                                     dsun_meters=current_wave_map.dsun.to('m').value)

        # Coordinate positions (HPC) with corresponding map data
        points = np.vstack((xx.ravel(), yy.ravel())).T
        values = np.asarray(deepcopy(current_wave_map.data)).ravel()

        # Solar rotation can push the points off disk and into areas that have
        # nans.  if this is the case, then griddata fails
        # Two solutions
        # 1 - replace all the nans with zeros, in order to get the code to run
        # 2 - the initial condition of zpp.ravel() >= 0 should be extended
        #     to make sure that only finite points are used.

        # 2D interpolation from origin grid to destination grid
        valid_points = np.logical_and(zpp.ravel() >= 0,
                                      np.isfinite(points[:, 0]),
                                      np.isfinite(points[:, 1]))
        grid = griddata(points[valid_points],
                        values[valid_points],
                        (hpcx_grid, hpcy_grid),
                        method="linear")
        transformed_wave_map = Map(grid, MapMeta(dict_header))
        transformed_wave_map.plot_settings = deepcopy(current_wave_map.plot_settings)
        # transformed_wave_map.name = current_wave_map.name
        # transformed_wave_map.meta['date-obs'] = current_wave_map.date
        wave_maps_transformed.append(transformed_wave_map)

    return Map(wave_maps_transformed, cube=True)


def transform2(params, wave_maps, verbose=False):
    """
    Transform raw data in HG' coordinates to HPC coordinates

    HG' = HG, except center at wave epicenter
    """
    solar_rotation_rate = params["rotation"]

    # Storage for the HPC version of the input maps
    wave_maps_transformed = []

    # The properties of this map are used in the transform
    smap = wave_maps[0]

    for hg_map in wave_maps:
        # Elapsed time
        td = parse_time(hg_map.date) - parse_time(smap.date)

        total_seconds = u.s * (td.microseconds + (td.seconds + td.days * 24.0 * 3600.0) * 10.0**6) / 10.0**6
        solar_rotation = total_seconds * solar_rotation_rate

        solar_information = {"hglt_obs": params["hglt_obs"],
                             "angle_rotated": solar_rotation}

        hpc_map = map_hg_to_hpc_rotate(hg_map,
                                       epi_lon=params["epi_lon"],
                                       epi_lat=params["epi_lat"],
                                       xbin=params["hpcx_bin"],
                                       ybin=params["hpcy_bin"],
                                       xnum=params["xnum"],
                                       ynum=params["ynum"],
                                       solar_information=solar_information)

        wave_maps_transformed.append(hpc_map)

    return Map(wave_maps_transformed, cube=True)


def noise_random(params, shape):
    """Return an ndarray of random noise"""
    
    noise_type = params.get("noise_type")
    noise_scale = params.get("noise_scale")
    noise_mean = params.get("noise_mean")
    noise_sdev = params.get("noise_sdev")
    
    if noise_type is None:
        noise = np.zeros(shape)
    else:
        if noise_type == "Normal":
            noise = noise_scale*np.random.normal(noise_mean, noise_sdev, shape)
        elif noise_type == "Poisson":
            noise = noise_scale*np.random.poisson(noise_mean, shape)
        else:
            noise = np.zeros(shape)
    
    return noise


def noise_structure(params, shape):
    """Return an ndarray of structured noise"""
    
    struct_type = params.get("struct_type")
    struct_scale = params.get("struct_scale")
    struct_num = params.get("struct_num")
    struct_seed = params.get("struct_seed")
    
    if struct_type is None:
        struct = np.zeros(shape)
    else:

        if struct_type == "Arcs":
            struct = np.zeros(shape)
            
            rsigma = 5
            
            xc = np.random.random_sample(struct_num)*shape[0]
            yc = np.random.random_sample(struct_num)*shape[1]
            xo = np.random.random_sample(struct_num)*shape[0]
            yo = np.random.random_sample(struct_num)*shape[1]
            halfangle = np.random.random_sample(struct_num)*np.pi/4.
            
            r0 = np.sqrt((xc-xo)**2+(yc-yo)**2)
            # theta0 = np.arctan2(yc-yo, xc-xo)
                        
            x0, y0 = np.mgrid[0:shape[0], 0:shape[1]]

            for index in xrange(struct_num):
                x = x0 + rsigma*(np.random.random_sample()-0.5)
                y = y0 + rsigma*(np.random.random_sample()-0.5)
                
                r = np.sqrt((x-xo[index])**2+(y-yo[index])**2)
                # theta = np.arctan2(y-yo[index], x-xo[index])
                
                theta = np.arccos(((x-xo[index])*(xc[index]-xo[index])+(y-yo[index])*(yc[index]-yo[index]))/(r*r0[index]))
                
                struct += struct_scale*1/np.sqrt(2*np.pi*rsigma**2)*np.exp(-((r-r0[index])/rsigma)**2/2.)*(theta<=halfangle[index])
            
        elif struct_type == "Random":
            struct = struct_scale*noise_random(params, shape)
        else:
            struct = np.zeros(shape)

    return struct


def add_noise(params, wave_maps, verbose=False):
    """
    Adds simulated noise to a list of maps
    """
    wave_maps_noise = []
    for current_wave_map in wave_maps:
        if verbose:
            print("  * Adding noise to map at " + str(current_wave_map.date))

        noise = noise_random(params, current_wave_map.data.shape)
        struct = noise_structure(params, current_wave_map.data.shape)

        noisy_wave_map = Map(current_wave_map.data + noise + struct,
                                       current_wave_map.meta)
        noisy_wave_map.plot_settings = deepcopy(current_wave_map.plot_settings)
        wave_maps_noise.append(noisy_wave_map)

    return Map(wave_maps_noise, cube=True)


def clean(params, wave_maps, verbose=False):
    """
    Cleans a list of maps
    """
    wave_maps_clean = []
    for current_wave_map in wave_maps:
        if verbose:
            print("  * Cleaning map at "+str(current_wave_map.date))

        data = np.asarray(current_wave_map.data)
        if params.get("clean_nans"):
            data[np.isnan(data)] = 0.
                
        cleaned_wave_map = Map(data, current_wave_map.meta)
        # cleaned_wave_map.name = current_wave_map.name
        cleaned_wave_map.meta['date-obs'] = current_wave_map.date
        cleaned_wave_map.plot_settings = deepcopy(current_wave_map.plot_settings)
        wave_maps_clean.append(cleaned_wave_map)

    return Map(wave_maps_clean, cube=True)


def simulate(params, max_steps, verbose=False, output=['finalmaps'],
             use_transform2=False):
    """
    Simulates wave in HPC coordinates with added noise
    """
    # Storage for the output
    answer = {}

    # Create each stage in the simulation
    if verbose:
        print('  * Creating raw HG data')
    raw = simulate_raw(params, max_steps, verbose=verbose)

    if verbose:
        print('  * Transforming HG to HPC data')
    if use_transform2:
        transformed = transform2(params, raw, verbose=verbose)
        if verbose:
            print('  * Using transform2')
    else:
        transformed = transform(params, raw, verbose=verbose)
        if verbose:
            print('  * Using transform')

    if verbose:
        print('  * Adding noise to HPC data')
    noise = add_noise(params, transformed, verbose=verbose)

    if verbose:
        print('  * Cleaning up HPC data')
    finalmaps = clean(params, noise, verbose=verbose)

    if 'raw' in output:
        answer['raw'] = raw

    if 'transformed' in output:
        answer['transformed'] = transformed

    if 'noise' in output:
        answer['noise'] = noise

    if 'finalmaps' in output:
        answer['finalmaps'] = finalmaps

    return answer
