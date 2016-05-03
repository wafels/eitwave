#
# Transform maps and mapcubes to and from heliographic and helioprojective
# Cartesian co-ordinate systems.
#

from copy import deepcopy
from scipy.interpolate import griddata
import numpy as np
import numpy.ma as ma
import astropy.units as u
from sunpy.map import Map, MapMeta
from sunpy import wcs


__authors__ = ["Steven Christe", "Jack Ireland"]
__email__ = ["steven.d.christe@nasa.gov", "jack.ireland@nasa.gov"]

#
# Has value 0.5 in original wave2d.py code
# This makes hpcx_min the left edge of the first bin
#
# Has value 1.0 in the original util.py code.
# This makes hpcx.min() the center of the first bin.
# This makes hpct.min() the center of the first bin.
#
crpix12_value_for_HPC = 1.0
#
# Has value 0.5 in original wave2d.py code
# This makes lon_min the left edge of the first bin
# This makes lat_min the left edge of the first bin
#
# Has value 1.0 in the original util.py code
#
#
crpix12_value_for_HG = 1.0


def mapcube_hpc_to_hg(hpc_mapcube, params, verbose=True, **kwargs):
    """ Unravel the maps in SunPy from HPC to HG co-ordinates.  The **kwargs
    get passed through."""
    new_maps = []
    for index, m in enumerate(hpc_mapcube):
        if verbose:
            print("Unraveling map %(#)i of %(n)i " % {'#': index + 1, 'n': len(hpc_mapcube)})
        hg_map = map_hpc_to_hg_rotate(m, epi_lon=params.get('epi_lon'),
                                         epi_lat=params.get('epi_lat'),
                                         lon_num=params.get('lon_num'),
                                         lat_num=params.get('lat_num'), **kwargs)
        new_maps.append(hg_map)
    return Map(new_maps, cube=True)


def mapcube_hg_to_hpc(hg_mapcube, params, verbose=True, **kwargs):
    """ Transform HG maps into HPC maps. """
    new_maps = []
    for index, m in enumerate(hg_mapcube):
        if verbose:
            print("Transforming back to heliocentric coordinates map %(#)i of %(n)i " % {'#': index+1, 'n': len(hg_mapcube)})
        hpc_map = map_hg_to_hpc_rotate(m, epi_lon=params.get('epi_lon'),
                                          epi_lat=params.get('epi_lat'),
                                          xnum=params.get('xnum'),
                                          ynum=params.get('ynum'), **kwargs)
        new_maps += [hpc_map]
    return Map(new_maps, cube=True)


def map_hpc_to_hg_rotate(m,
                         epi_lon=0*u.degree, epi_lat=90*u.degree,
                         lon_bin=1*u.degree, lat_bin=1*u.degree,
                         lon_num=None, lat_num=None, **kwargs):
    """
    Transform raw data in HPC coordinates to HG' coordinates

    HG' = HG, except center at wave epicenter
    """
    x, y = wcs.convert_pixel_to_data([m.data.shape[1], m.data.shape[0]],
                                     [m.scale.x.value, m.scale.y.value],
                                     [m.reference_pixel.x.value, m.reference_pixel.y.value],
                                     [m.reference_coordinate.x.value, m.reference_coordinate.y.value])

    hccx, hccy, hccz = wcs.convert_hpc_hcc(x,
                                           y,
                                           angle_units=m.units.x,
                                           dsun_meters=m.dsun.to('meter').value,
                                           z=True)

    rot_hccz, rot_hccx, rot_hccy = euler_zyz((hccz,
                                              hccx,
                                              hccy),
                                             (0.,
                                              epi_lat.to('degree').value-90.,
                                              -epi_lon.to('degree').value))

    lon_map, lat_map = wcs.convert_hcc_hg(rot_hccx,
                                          rot_hccy,
                                          b0_deg=m.heliographic_latitude.to('degree').value,
                                          l0_deg=m.heliographic_longitude.to('degree').value,
                                          z=rot_hccz)

    lon_range = (np.nanmin(lon_map), np.nanmax(lon_map))
    lat_range = (np.nanmin(lat_map), np.nanmax(lat_map))

    # This method results in a set of lons and lats that in general does not
    # exactly span the range of the data.
    # lon = np.arange(lon_range[0], lon_range[1], lon_bin)
    # lat = np.arange(lat_range[0], lat_range[1], lat_bin)

    # This method gives a set of lons and lats that exactly spans the range of
    # the data at the expense of having to define values of cdelt1 and cdelt2
    if lon_num is None:
        cdelt1 = lon_bin.to('degree').value
        lon = np.arange(lon_range[0], lon_range[1], cdelt1)
    else:
        nlon = lon_num.to('pixel').value
        cdelt1 = (lon_range[1] - lon_range[0]) / (1.0*nlon - 1.0)
        lon = np.linspace(lon_range[0], lon_range[1], num=nlon)

    if lat_num is None:
        cdelt2 = lat_bin.to('degree').value
        lat = np.arange(lat_range[0], lat_range[1], cdelt2)
    else:
        nlat = lat_num.to('pixel').value
        cdelt2 = (lat_range[1] - lat_range[0]) / (1.0*nlat - 1.0)
        lat = np.linspace(lat_range[0], lat_range[1], num=nlat)

    # Create the grid
    x_grid, y_grid = np.meshgrid(lon, lat)

    ng_xyz = wcs.convert_hg_hcc(x_grid,
                                y_grid,
                                b0_deg=m.heliographic_latitude.to('degree').value,
                                l0_deg=m.heliographic_longitude.to('degree').value,
                                z=True)

    ng_zp, ng_xp, ng_yp = euler_zyz((ng_xyz[2],
                                     ng_xyz[0],
                                     ng_xyz[1]),
                                    (epi_lon.to('degree').value,
                                     90.-epi_lat.to('degree').value,
                                     0.))

    # The function ravel flattens the data into a 1D array
    points = np.vstack((lon_map.ravel(), lat_map.ravel())).T
    values = np.array(m.data).ravel()

    # Get rid of all of the bad (nan) indices (i.e. those off of the sun)
    index = np.isfinite(points[:, 0]) * np.isfinite(points[:, 1])
    # points = np.vstack((points[index,0], points[index,1])).T
    points = points[index]
    values = values[index]

    newdata = griddata(points, values, (x_grid, y_grid), **kwargs)
    newdata[ng_zp < 0] = np.nan

    dict_header = {
        'CDELT1': cdelt1,
        'NAXIS1': len(lon),
        'CRVAL1': lon.min(),
        'CRPIX1': crpix12_value_for_HG,
        'CRPIX2': crpix12_value_for_HG,
        'CUNIT1': "deg",
        'CTYPE1': "HG",
        'CDELT2': cdelt2,
        'NAXIS2': len(lat),
        'CRVAL2': lat.min(),
        'CUNIT2': "deg",
        'CTYPE2': "HG",
        'DATE_OBS': m.meta['date-obs'],
        'DSUN_OBS': m.dsun.to('m').value,
        "CRLN_OBS": m.carrington_longitude.to('degree').value,
        "HGLT_OBS": m.heliographic_latitude.to('degree').value,
        "HGLN_OBS": m.heliographic_longitude.to('degree').value,
        'EXPTIME': m.exposure_time.to('s').value
    }

    # Find out where the non-finites are
    mask = np.logical_not(np.isfinite(newdata))

    # Return a masked array is appropriate
    if mask is not None:
        return Map(newdata, MapMeta(dict_header))
    else:
        return Map(ma.array(newdata, mask=mask), MapMeta(dict_header))


def map_hg_to_hpc_rotate(m,
                         epi_lon=90*u.degree, epi_lat=0*u.degree,
                         xbin=2.4*u.arcsec, ybin=2.4*u.arcsec,
                         xnum=None, ynum=None,
                         solar_information=None, **kwargs):
    """
    Transform raw data in HG' coordinates to HPC coordinates

    HG' = HG, except center at wave epicenter
    """

    # Origin grid, HG'
    lon_grid, lat_grid = wcs.convert_pixel_to_data([m.data.shape[1], m.data.shape[0]],
                                                   [m.scale.x.value, m.scale.y.value],
                                                   [m.reference_pixel.x.value, m.reference_pixel.y.value],
                                                   [m.reference_coordinate.x.value, m.reference_coordinate.y.value])

    # Origin grid, HG' to HCC'
    # HCC' = HCC, except centered at wave epicenter
    x, y, z = wcs.convert_hg_hcc(lon_grid, lat_grid,
                                 b0_deg=m.heliographic_latitude.to('degree').value,
                                 l0_deg=m.carrington_longitude.to('degree').value,
                                 z=True)

    # Origin grid, HCC' to HCC''
    # Moves the wave epicenter to initial conditions
    # HCC'' = HCC, except assuming that HGLT_OBS = 0
    zpp, xpp, ypp = euler_zyz((z,
                               x,
                               y),
                              (epi_lon.to('degree').value,
                               90.-epi_lat.to('degree').value,
                               0.))

    # Add in a solar rotation.  Useful when creating simulated HPC data from
    # HG data.  This code was adapted from the wave simulation code of the
    # AWARE project.
    if solar_information is not None:
        hglt_obs = solar_information['hglt_obs'].to('degree').value
        solar_rotation_value = solar_information['angle_rotated'].to('degree').value
        #print(hglt_obs, solar_rotation_value)
        #print('before', zpp, xpp, ypp)
        zpp, xpp, ypp = euler_zyz((zpp,
                                   xpp,
                                   ypp),
                                  (0.,
                                   hglt_obs,
                                   solar_rotation_value))
        #print('after', zpp, xpp, ypp)
    # Origin grid, HCC to HPC (arcsec)
    # xx, yy = wcs.convert_hcc_hpc(current_wave_map.header, xpp, ypp)
    xx, yy = wcs.convert_hcc_hpc(xpp, ypp,
                                 dsun_meters=m.dsun.to('meter').value)

    # Destination HPC grid
    hpcx_range = (np.nanmin(xx), np.nanmax(xx))
    hpcy_range = (np.nanmin(yy), np.nanmax(yy))

    if xnum is None:
        cdelt1 = xbin.to('arcsec').value
        hpcx = np.arange(hpcx_range[0], hpcx_range[1], cdelt1)
    else:
        nx = xnum.to('pixel').value
        cdelt1 = (hpcx_range[1] - hpcx_range[0]) / (1.0*nx - 1.0)
        hpcx = np.linspace(hpcx_range[1], hpcx_range[0], num=nx)

    if ynum is None:
        cdelt2 = ybin.to('arcsec').value
        hpcy = np.arange(hpcy_range[0], hpcy_range[1], cdelt2)
    else:
        ny = ynum.to('pixel').value
        cdelt2 = (hpcy_range[1] - hpcy_range[0]) / (1.0*ny - 1.0)
        hpcy = np.linspace(hpcy_range[1], hpcy_range[0], num=ny)

    # Calculate the grid mesh
    newgrid_x, newgrid_y = np.meshgrid(hpcx, hpcy)

    dict_header = {
        "CDELT1": cdelt1,
        "NAXIS1": len(hpcx),
        "CRVAL1": hpcx.min(),
        "CRPIX1": crpix12_value_for_HPC,
        "CUNIT1": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CDELT2": cdelt2,
        "NAXIS2": len(hpcy),
        "CRVAL2": hpcy.min(),
        "CRPIX2": crpix12_value_for_HPC,
        "CUNIT2": "arcsec",
        "CTYPE2": "HPLT-TAN",
        "HGLT_OBS": m.heliographic_latitude.to('degree').value,  # 0.0
        # "HGLN_OBS": 0.0,
        "CRLN_OBS": m.carrington_longitude.to('degree').value,  # 0.0
        'DATE_OBS': m.meta['date-obs'],
        'DSUN_OBS': m.dsun.to('m').value,
        'EXPTIME': m.exposure_time.to('s').value
    }

    # Coordinate positions (HPC) with corresponding map data
    points = np.vstack((xx.ravel(), yy.ravel())).T
    values = np.asarray(deepcopy(m.data)).ravel()

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
    # 2D interpolation from origin grid to destination grid
    grid = griddata(points[valid_points],
                    values[valid_points],
                    (newgrid_x, newgrid_y), **kwargs)

    # Find out where the non-finites are
    mask = np.logical_not(np.isfinite(grid))

    # Return a masked array is appropriate
    if mask is not None:
        return Map(grid, MapMeta(dict_header))
    else:
        return Map(ma.array(grid, mask=mask), MapMeta(dict_header))


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
