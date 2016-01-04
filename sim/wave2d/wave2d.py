"""
Simulates a wave
"""

from __future__ import absolute_import

__all__ = ["simulate", "simulate_raw", "transform", "add_noise"]

__authors__ = ["Albert Shih"]
__email__ = "albert.y.shih@nasa.gov"

import numpy as np
import sunpy
import sunpy.map
import astropy.units as u
import sunpy.map as Map
import sunpy.sun.sun as sun
import datetime
from sunpy.time import parse_time
from scipy.special import ndtr
from scipy.interpolate import griddata

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


def euler_zyz(xyz, angles):
    """
    Rotation with Euler angles defined in the ZYZ convention with left-handed
    positive sign convention.
    
    Parameters
    ----------
        xyz : tuple of ndarrays
            Input coordinates
        angles : tuple of scalars
            Angular rotations are applied in the following order
            * angles[2] is the angle CCW around Z axis (intrinsic rotation)
            * angles[1] is the angle CCW around Y axis (polar angle)
            * angles[0] is the angle CCW around Z axis (azimuth angle)
    
    Returns
    -------
        X, Y, Z : ndarray
            Output coordinates
    
    Notes
    -----
    angles = (phi, theta, psi) inverts angles = (-psi, -theta, -phi)

    References
    ----------
    https://en.wikipedia.org/wiki/Euler_angles#Matrix_orientation
        ("Left-handed positive sign convention", "ZYZ")
    
    Examples
    --------
    >>> wave2d.euler_zyz((np.array([1,0,0]),
                          np.array([0,1,0]),
                          np.array([0,0,1])),
                         (45,45,0))
    (array([ 0.5       , -0.70710678,  0.5       ]),
     array([ 0.5       ,  0.70710678,  0.5       ]),
     array([-0.70710678,  0.        ,  0.70710678]))
    """
    c1 = np.cos(np.deg2rad(angles[0]))
    s1 = np.sin(np.deg2rad(angles[0]))
    c2 = np.cos(np.deg2rad(angles[1]))
    s2 = np.sin(np.deg2rad(angles[1]))
    c3 = np.cos(np.deg2rad(angles[2]))
    s3 = np.sin(np.deg2rad(angles[2]))
    x = (c1*c2*c3-s1*s3)*xyz[0]+(-c3*s1-c1*c2*s3)*xyz[1]+(c1*s2)*xyz[2]
    y = (+c1*s3+c2*c3*s1)*xyz[0]+(c1*c3-c2*s1*s3)*xyz[1]+(s1*s2)*xyz[2]
    z = (-c3*s2)*xyz[0]+(s2*s3)*xyz[1]+(c2)*xyz[2]
    return x, y, z


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
    speed_coeff = prep_coeff(params["speed"])

    lat_min = params["lat_min"].to('degree').value
    lat_max = params["lat_max"].to('degree').value
    lat_bin = params["lat_bin"].to('degree').value
    lon_min = params["lon_min"].to('degree').value
    lon_max = params["lon_max"].to('degree').value
    lon_bin = params["lon_bin"].to('degree').value

    #This roundabout approach recalculates lat_bin and lon_bin to produce
    #equally sized bins to exactly span the min/max ranges
    lat_num = int(round((lat_max-lat_min)/lat_bin))
    lat_edges, lat_bin = np.linspace(lat_min, lat_max, lat_num+1,
                                     retstep=True)
    lon_num = int(round((lon_max-lon_min)/lon_bin))
    lon_edges, lon_bin = np.linspace(lon_min, lon_max, lon_num+1,
                                     retstep=True)
    
    #Propagates from 90. down to lat_min, irrespective of lat_max
    p = np.poly1d([speed_coeff[2]/3., speed_coeff[1]/2., speed_coeff[0],
                   -(90.-lat_min)])
    #p = np.poly1d([0.0, speed_coeff[1], speed_coeff[2]/2.,
    #               -(90.-lat_min)])
    #Will fail if wave does not propogate all the way to lat_min
    #duration = p.r[np.logical_and(p.r.real > 0, p.r.imag == 0)][0]
    
    #steps = int(duration/cadence)+1
    #if steps > params["max_steps"]:
    #    steps = params["max_steps"]
    
    #Maybe used np.poly1d() instead to do the polynomial calculation?
    time = np.arange(steps)*cadence
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

    wave_maps = []
    
    dict_header = {
        "CDELT1": lon_bin,
        "NAXIS1": lon_num,
        "CRVAL1": lon_min,
        "CRPIX1": 0.5, #this makes lon_min the left edge of the first bin
        "CUNIT1": "deg",
        "CTYPE1": "HG",
        "CDELT2": lat_bin,
        "NAXIS2": lat_num,
        "CRVAL2": lat_min,
        "CRPIX2": 0.5, #this makes lat_min the left edge of the first bin
        "CUNIT2": "deg",
        "CTYPE2": "HG",
        "HGLT_OBS": 0,
        "HGLN_OBS": 0,
        "DSUN_OBS": sun.sunearth_distance(BASE_DATE.strftime(BASE_DATE_FORMAT)),
        "DATE_OBS": BASE_DATE.strftime(BASE_DATE_FORMAT),
        "EXPTIME": 1.0
    }

    if verbose:
        print("Simulating "+str(steps)+" raw maps")

    for istep in xrange(steps):
        dict_header['DATE_OBS'] = (BASE_DATE + datetime.timedelta(seconds=istep * cadence)).strftime(BASE_DATE_FORMAT)

        header = sunpy.map.MapMeta(dict_header)

        # Gaussian profile in longitudinal direction
        # Does not take into account spherical geometry
        # (i.e., change in area element)
        if wave_thickness[istep] <= 0:
            print("ERROR: wave thickness is non-physical!")
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
        
        wave_maps += [sunpy.map.Map(wave, header)]
        #wave_maps[istep].name = "Simulation"
        # wave_maps[istep].meta['date-obs'] = parse_time("2011-11-11")+datetime.timedelta(0, istep*cadence)
    
    return sunpy.map.Map(wave_maps, cube=True)


def transform(params, wave_maps, verbose=False):
    """
    Transform raw data in HG' coordinates to HPC coordinates
    
    HG' = HG, except center at wave epicenter
    """
    cadence = params["cadence"]
    hglt_obs = params["hglt_obs"]
    rotation = params["rotation"]
    
    epi_lat = params["epi_lat"]
    epi_lon = params["epi_lon"]
    
    hpcx_min = params["hpcx_min"]
    hpcx_max = params["hpcx_max"]
    hpcx_bin = params["hpcx_bin"]
    hpcy_min = params["hpcy_min"]
    hpcy_max = params["hpcy_max"]
    hpcy_bin = params["hpcy_bin"]
    
    hpcx_num = int(round((hpcx_max-hpcx_min)/hpcx_bin))
    hpcy_num = int(round((hpcy_max-hpcy_min)/hpcy_bin))
    
    wave_maps_transformed = []
    
    dict_header = {
        "CDELT1": hpcx_bin.to('arcsec').value,
        "NAXIS1": hpcx_num,
        "CRVAL1": hpcx_min.to('arcsec').value,
        "CRPIX1": 0.5,  # this makes hpcx_min the left edge of the first bin
        "CUNIT1": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CDELT2": hpcy_bin.to('arcsec').value,
        "NAXIS2": hpcy_num,
        "CRVAL2": hpcy_min.to('arcsec').value,
        "CRPIX2": 0.5,  # this makes hpcy_min the left edge of the first bin
        "CUNIT2": "arcsec",
        "CTYPE2": "HPLT-TAN",
        "HGLT_OBS": hglt_obs.to('degree').value,
        "HGLN_OBS": 0,
        "DSUN_OBS": sun.sunearth_distance(BASE_DATE.strftime(BASE_DATE_FORMAT)),
        "DATE_OBS": BASE_DATE.strftime(BASE_DATE_FORMAT),
        "EXPTIME": 1.0
    }
    
    header = sunpy.map.MapMeta(dict_header)
    
    start_date = wave_maps[0].date
    
    # Origin grid, HG'
    print wave_maps[0].scale.x.value
    lon_grid, lat_grid = sunpy.wcs.convert_pixel_to_data([wave_maps[0].data.shape[1], wave_maps[0].data.shape[0]],
                                                         [wave_maps[0].scale.x.value, wave_maps[0].scale.y.value],
                                                         [wave_maps[0].reference_pixel.x.value, wave_maps[0].reference_pixel.y.value],
                                                         [wave_maps[0].reference_coordinate.x.value, wave_maps[0].reference_coordinate.y.value])
    
    # Origin grid, HG' to HCC'
    # HCC' = HCC, except centered at wave epicenter
    x, y, z = sunpy.wcs.convert_hg_hcc(lon_grid, lat_grid,
                                       wave_maps[0].heliographic_latitude,
                                       wave_maps[0].carrington_longitude.to('degree').value,
                                       z=True)
    
    # Origin grid, HCC' to HCC''
    # Moves the wave epicenter to initial conditions
    # HCC'' = HCC, except assuming that HGLT_OBS = 0
    zxy_p = euler_zyz((z, x, y), (epi_lon.to('degree').value, 90.-epi_lat.to('degree').value, 0.))
    
    # Destination HPC grid
    hpcx_grid, hpcy_grid = sunpy.wcs.convert_pixel_to_data([header['NAXIS1'], header['NAXIS2']],
                                                           [header['CDELT1'], header['CDELT2']],
                                                           [header['CRPIX1'], header['CRPIX2']],
                                                           [header['CRVAL1'], header['CRVAL2']])

    for icwm, current_wave_map in enumerate(wave_maps):
        dict_header['DATE_OBS'] = (BASE_DATE + datetime.timedelta(seconds=icwm * cadence)).strftime(BASE_DATE_FORMAT)
        header = sunpy.map.MapMeta(dict_header)

        # Origin grid, HCC'' to HCC
        # Moves the observer to HGLT_OBS and adds rigid solar rotation
        td = parse_time(current_wave_map.date) - parse_time(start_date)
        total_seconds = u.s * (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6
        zpp, xpp, ypp = euler_zyz(zxy_p, (0., hglt_obs.to('degree').value, (total_seconds*rotation).to('degree').value))
        
        # Origin grid, HCC to HPC (arcsec)
        xx, yy = sunpy.wcs.convert_hcc_hpc(xpp, ypp,
                                           current_wave_map.dsun)
        
        # Coordinate positions (HPC) with corresponding map data
        points = np.vstack((xx.ravel(), yy.ravel())).T
        values = np.asarray(current_wave_map.data).ravel()

        # 2D interpolation from origin grid to destination grid
        grid = griddata(points[zpp.ravel() >= 0], values[zpp.ravel() >= 0],
                        (hpcx_grid, hpcy_grid), method="linear")
        transformed_wave_map = sunpy.map.Map(grid, header)
        # transformed_wave_map.name = current_wave_map.name
        # transformed_wave_map.meta['date-obs'] = current_wave_map.date
        wave_maps_transformed.append(transformed_wave_map)

    return sunpy.map.Map(wave_maps_transformed, cube=True)


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
            print("Adding noise to map at " + str(current_wave_map.date))

        noise = noise_random(params, current_wave_map.data.shape)
        struct = noise_structure(params, current_wave_map.data.shape)

        noisy_wave_map = sunpy.map.Map(current_wave_map.data + noise + struct,
                                       current_wave_map.meta)
        # noisy_wave_map.name = current_wave_map.name
        noisy_wave_map.meta['date-obs'] = current_wave_map.date
        wave_maps_noise.append(noisy_wave_map)

    return sunpy.map.Map(wave_maps_noise, cube=True)


def clean(params, wave_maps, verbose=False):
    """
    Cleans a list of maps
    """
    wave_maps_clean = []
    for current_wave_map in wave_maps:
        if verbose:
            print("Cleaning map at "+str(current_wave_map.date))

        data = np.asarray(current_wave_map.data)
        if params.get("clean_nans"):
            data[np.isnan(data)] = 0.
                
        cleaned_wave_map = sunpy.map.Map(data, current_wave_map.meta)
        # cleaned_wave_map.name = current_wave_map.name
        cleaned_wave_map.meta['date-obs'] = current_wave_map.date
        wave_maps_clean.append(cleaned_wave_map)

    return sunpy.map.Map(wave_maps_clean, cube=True)


def simulate(params, max_steps, verbose=False, output=['finalmaps']):
    """
    Simulates wave in HPC coordinates with added noise
    """
    # Storage for the output
    answer = {}
    # Create each stage in the simulation
    raw = simulate_raw(params, max_steps, verbose=verbose)
    transformed = transform(params, raw, verbose=verbose)
    noise = add_noise(params, transformed, verbose=verbose)
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
