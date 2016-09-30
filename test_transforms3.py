#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import astropy.units as u

# Not used in the module but useful on the command line
import numpy as np

import matplotlib.pyplot as plt
from sunpy.map import Map

from map_hpc_hg_transforms import map_hg_to_hpc_rotate, map_hpc_to_hg_rotate, euler_zyz


from copy import deepcopy
from scipy.interpolate import griddata
import numpy as np
import numpy.ma as ma
import astropy.units as u
from sunpy.map import Map, MapMeta
from sunpy import wcs


plt.ion()


# Select the wave
file_path = '/Users/ireland/sunpy/data/aia_lev1_304a_2013_03_04t01_00_07_12z_image_lev1.fits'
m = Map(file_path).superpixel((4, 4)*u.pix)

ny = m.data.shape[0]
nx = m.data.shape[1]
dd = 25
aa = 200
bb = 800

m.data[ny/2 - dd: ny/2 + dd, 200-dd:200+dd] = 0.0
m.data[ny/2 - dd: ny/2 + dd, 800-dd:800+dd] = 0.33 * m.data.max()
m.data[200-dd:200+dd, nx/2 - dd: nx/2 + dd] = 0.66 * m.data.max()
m.data[800-dd:800+dd, nx/2 - dd: nx/2 + dd] = 1.00 * m.data.max()

m.peek()
# Unraveling parameters used to convert HPC image data to HG data
along_wavefront_sampling = 1
perpendicular_to_wavefront_sampling = 1
transform_hpc2hg_parameters = {'epi_lon': 00.0*u.degree,
                               'epi_lat': 0.0*u.degree,
                               'lon_num': 360*along_wavefront_sampling*u.pixel,
                               'lat_num': 720*perpendicular_to_wavefront_sampling*u.pixel}

hg = map_hpc_to_hg_rotate(m,
                          epi_lon=transform_hpc2hg_parameters['epi_lon'],
                          epi_lat=transform_hpc2hg_parameters['epi_lat'],
                          lon_num=transform_hpc2hg_parameters['lon_num'],
                          lat_num=transform_hpc2hg_parameters['lat_num'],
                          method='linear')

fig, ax = plt.subplots()
im = hg.plot()
ax.set_autoscale_on(False)
ax.set_title('Custom plot without WCSAxes')
plt.colorbar()
plt.show()
# Unraveling parameters used to convert HG image data to HPC data
transform_hg2hpc_parameters = {'epi_lon': transform_hpc2hg_parameters['epi_lon'],
                               'epi_lat': transform_hpc2hg_parameters['epi_lat'],
                               'xnum': 1024*u.pixel,
                               'ynum': 256*u.pixel}

hpc = map_hg_to_hpc_rotate(hg,
                           epi_lon=transform_hg2hpc_parameters['epi_lon'],
                           epi_lat=transform_hg2hpc_parameters['epi_lat'],
                           xnum=transform_hg2hpc_parameters['xnum'],
                           ynum=transform_hg2hpc_parameters['ynum'])
hpc.peek(aspect='auto')

#
# copied from map_hg_to_hpc_rotate
#
xnum = transform_hg2hpc_parameters['xnum']
ynum = transform_hg2hpc_parameters['ynum']
epi_lon = transform_hg2hpc_parameters['epi_lon']
epi_lat = transform_hg2hpc_parameters['epi_lat']


# Origin grid, HG'
lon_grid, lat_grid = wcs.convert_pixel_to_data(
    [hg.data.shape[1], hg.data.shape[0]],
    [hg.scale.x.value, hg.scale.y.value],
    [hg.reference_pixel.x.value, hg.reference_pixel.y.value],
    [hg.reference_coordinate.x.value, hg.reference_coordinate.y.value])

# Origin grid, HG' to HCC'
# HCC' = HCC, except centered at wave epicenter
x, y, z = wcs.convert_hg_hcc(lon_grid, lat_grid,
                             b0_deg=hg.heliographic_latitude.to('degree').value,
                             l0_deg=hg.carrington_longitude.to('degree').value,
                             z=True)

# Origin grid, HCC' to HCC''
# Moves the wave epicenter to initial conditions
# HCC'' = HCC, except assuming that HGLT_OBS = 0
zpp, xpp, ypp = euler_zyz((z,
                           x,
                           y),
                          (0,
                           90.0 - epi_lon.to('degree').value,
                           epi_lat.to('degree').value))

# Add in a solar rotation.  Useful when creating simulated HPC data from
# HG data.  This code was adapted from the wave simulation code of the
# AWARE project.
"""
if solar_information is not None:
    hglt_obs = solar_information['hglt_obs'].to('degree').value
    solar_rotation_value = solar_information['angle_rotated'].to(
        'degree').value
    # print(hglt_obs, solar_rotation_value)
    # print('before', zpp, xpp, ypp)
    zpp, xpp, ypp = euler_zyz((zpp,
                               xpp,
                               ypp),
                              (0.,
                               hglt_obs,
                               solar_rotation_value))
    # print('after', zpp, xpp, ypp)
"""
# Origin grid, HCC to HPC (arcsec)
# xx, yy = wcs.convert_hcc_hpc(current_wave_map.header, xpp, ypp)
xx, yy = wcs.convert_hcc_hpc(xpp, ypp,
                             dsun_meters=hg.dsun.to('meter').value)

# Destination HPC grid
hpcx_range = (np.nanmin(xx), np.nanmax(xx))
hpcy_range = (np.nanmin(yy), np.nanmax(yy))

if xnum is None:
    cdelt1 = xbin.to('arcsec').value
    hpcx = np.arange(hpcx_range[0], hpcx_range[1], cdelt1)
else:
    nx = xnum.to('pixel').value
    cdelt1 = (hpcx_range[1] - hpcx_range[0]) / (1.0 * nx - 1.0)
    hpcx = np.linspace(hpcx_range[1], hpcx_range[0], num=nx)

if ynum is None:
    cdelt2 = ybin.to('arcsec').value
    hpcy = np.arange(hpcy_range[0], hpcy_range[1], cdelt2)
else:
    ny = ynum.to('pixel').value
    cdelt2 = (hpcy_range[1] - hpcy_range[0]) / (1.0 * ny - 1.0)
    hpcy = np.linspace(hpcy_range[1], hpcy_range[0], num=ny)

# Calculate the grid mesh
newgrid_x, newgrid_y = np.meshgrid(hpcx, hpcy)

#
# CRVAL1,2 and CRPIX1,2 are calculated so that the co-ordinate system is
# at the center of the image
# Note that crpix[] counts pixels starting at 1
crpix1 = 1 + hpcx.size // 2
crval1 = hpcx[crpix1 - 1]
crpix2 = 1 + hpcy.size // 2
crval2 = hpcy[crpix2 - 1]
dict_header = {
    "CDELT1": cdelt1,
    "NAXIS1": len(hpcx),
    "CRVAL1": crval1,
    "CRPIX1": crpix1,
    "CUNIT1": "arcsec",
    "CTYPE1": "HPLN-TAN",
    "CDELT2": cdelt2,
    "NAXIS2": len(hpcy),
    "CRVAL2": crval2,
    "CRPIX2": crpix2,
    "CUNIT2": "arcsec",
    "CTYPE2": "HPLT-TAN",
    "HGLT_OBS": hg.heliographic_latitude.to('degree').value,  # 0.0
    # "HGLN_OBS": 0.0,
    "CRLN_OBS": hg.carrington_longitude.to('degree').value,  # 0.0
    'DATE_OBS': hg.meta['date-obs'],
    'DSUN_OBS': hg.dsun.to('m').value,
    'EXPTIME': hg.exposure_time.to('s').value
}

# Coordinate positions (HPC) with corresponding map data
points = np.vstack((xx.ravel(), yy.ravel())).T
values = np.asarray(deepcopy(hg.data)).ravel()

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
                (newgrid_x, newgrid_y))

# Find out where the non-finites are
mask = np.logical_not(np.isfinite(grid))

# Return a masked array is appropriate
if mask is None:
    hpc2 = Map(grid, MapMeta(dict_header))
else:
    hpc2 = Map(ma.array(grid, mask=mask), MapMeta(dict_header))

hpc2.plot_settings = hg.plot_settings
hpc2.peek(aspect='auto')
