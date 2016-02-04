#
# Load in multiple noisy realizations of a given simulated wave, and
# characterize the performance of AWARE on detecting the wave
#
import os
import astropy.units as u

# Not used in the module but useful on the command line
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
# Main AWARE processing and detection code
import util

# For testing the transformation
from copy import deepcopy
from scipy.interpolate import griddata
import numpy as np
from sunpy.map import Map, MapMeta
from sunpy import wcs

# Wave simulation code
import test_wave2d

# Simulated wave parameters
import swave_params

crpix12_value_for_HPC = 1.0

# Simulated data
# TODO - run the same analysis on multiple noisy realizations of the simulated
# TODO - data. Generate average result plots over the multiple realizations
# TODO - either as relative error or absolute error as appropriate.
# TODO - First mode: use multiple noisy realizations of the same model.
# TODO - Second mode: choose a bunch of simulated data parameters, and then
# TODO - randomly select values for them (within reason).
# TODO - The recovered parameters should be reasonably close to the simulated
# TODO - parameters.
#
#
random_seed = 42
np.random.seed(random_seed)

# Select the wave
#example = 'wavenorm4_slow'
example = 'no_noise_no_solar_rotation_slow_360'

# What type of output do we want to analyze
mctype = 'finalmaps'

# Number of images
max_steps = 2

# Accumulation in the time direction
accum = 2

# Summing in the spatial directions
spatial_summing = [4, 4]*u.pix

# Radii of the morphological operations
radii = [[5, 5], [11, 11], [22, 22]]

# Position measuring choices
position_choice = 'average'
error_choice = 'width'

# Degree of polynomial to fit
n_degree = 1

# Unraveling factor
unraveling_factor = 4.0

# Output directory
output = '~/eitwave/'

# Output types
otypes = ['img', 'pkl']

# RANSAC
ransac_kwargs = {"random_state": random_seed}

# Special designation: an extra description added to the file and directory
# names in order to differentiate between experiments on the same example wave.
special_designation = ''

# Output directories and filename
odir = os.path.expanduser(output)
otypes_dir = {}
otypes_filename = {}

# Morphological radii
sradii = ''
for r in radii:
    for v in r:
        sradii = sradii + str(v) + '_'
sradii = sradii[0: -1]


# Load in the simulated wave params
simulated_wave_parameters = swave_params.waves(lon_start=-180 * u.degree + 0.0 * u.degree)[example]

# Unraveling parameters used to convert HPC image data to HG data
unraveling_hpc2hg_parameters = {'lon_bin': 1.0*u.degree,
                                'lat_bin': 1.0*u.degree,
                                'epi_lon': 0.0*u.degree,
                                'epi_lat': 0.0*u.degree,
                                'lon_num': 200*u.pixel,
                                'lat_num': 300*u.pixel}


# Unraveling parameters used to convert HG image data to HPC data
unraveling_hg2hpc_parameters = {'epi_lon': simulated_wave_parameters['epi_lon'],
                                'epi_lat': simulated_wave_parameters['epi_lat'],
                                'xnum': 800*u.pixel,
                                'ynum': 800*u.pixel}


# Storage for the results
results = []

# Go through all the test waves, and apply AWARE.
# Let the user which trial is happening
print('\nSimulating %s ' % example)
print(' - position choice = %s' % position_choice)
print(' - error choice = %s' % error_choice)
print(' - unraveling factor = %f' % unraveling_factor)

# Simulate the wave and return a dictionary
out = test_wave2d.simulate_wave2d(params=simulated_wave_parameters, max_steps=max_steps,
                                  verbose=True, output=['raw', 'transformed', 'noise', 'finalmaps'])

# Get a raw HG map
map_index = 0
raw_hg_map = out['raw'][map_index]

# Get the corresponding transformed map
transformed_hpc_map = out['transformed'][map_index]

#
# Test the util conversion of a HG map to a HPC map
#
raw_hg_map_converted_to_hpc = util.map_hg_to_hpc_rotate(raw_hg_map,
                                                        epi_lon=unraveling_hg2hpc_parameters['epi_lon'],
                                                        epi_lat=unraveling_hg2hpc_parameters['epi_lat'],
                                                        xnum=unraveling_hg2hpc_parameters['xnum'],
                                                        ynum=unraveling_hg2hpc_parameters['ynum'])

# Extract the central portion of each map for comparison purposes.  When
# xnum and ynum above are set to 800, the two extracted maps have the same
# size in terms of pixels
extract_range = (-60, 60)*u.arcsec
smap1 = raw_hg_map_converted_to_hpc. submap(extract_range, extract_range)
smap2 = transformed_hpc_map.submap(extract_range, extract_range)


# Test the round trip conversion HG -> HPC -> HG using util
print('Testing round trip conversion HG -> HPC -> HG using util')
hg2hpc = util.map_hg_to_hpc_rotate(raw_hg_map,
                                   epi_lon=simulated_wave_parameters['epi_lon'],
                                   epi_lat=simulated_wave_parameters['epi_lat'],
                                   xnum=unraveling_hg2hpc_parameters['xnum'],
                                   ynum=unraveling_hg2hpc_parameters['ynum'])

hg2hpc2hg = util.map_hpc_to_hg_rotate(hg2hpc,
                                      epi_lon=unraveling_hpc2hg_parameters['epi_lon'],
                                      epi_lat=unraveling_hpc2hg_parameters['epi_lat'],
                                      lon_num=unraveling_hpc2hg_parameters['lon_num'],
                                      lat_num=unraveling_hpc2hg_parameters['lat_num'])


# Test the round trip conversion HPC -> HG -> HPC
print('Testing round trip conversion HPC -> HG -> HPC using util')
hpc2hg = util.map_hpc_to_hg_rotate(raw_hg_map_converted_to_hpc,
                                   epi_lon=unraveling_hpc2hg_parameters['epi_lon'],
                                   epi_lat=unraveling_hpc2hg_parameters['epi_lat'],
                                   lon_num=10*unraveling_hpc2hg_parameters['lon_num'],
                                   lat_num=10*unraveling_hpc2hg_parameters['lat_num'])

hpc2hghpc = util.map_hg_to_hpc_rotate(hpc2hg,
                                   epi_lon=simulated_wave_parameters['epi_lon'],
                                   epi_lat=simulated_wave_parameters['epi_lat'],
                                   xnum=unraveling_hg2hpc_parameters['xnum'],
                                   ynum=unraveling_hg2hpc_parameters['ynum'])



"""
# Test the round trip HPC -> HG -> HPC using util.
fmap = out['finalmaps'][map_index]
fmap2hg = util.map_hpc_to_hg_rotate(fmap,
                                    epi_lon=unraveling_hpc2hg_parameters['epi_lon'],
                                    epi_lat=unraveling_hpc2hg_parameters['epi_lat'],
                                    lon_bin=unraveling_hpc2hg_parameters['lon_bin'],
                                    lat_bin=unraveling_hpc2hg_parameters['lat_bin'],
                                    lon_num=unraveling_hpc2hg_parameters['lon_num'],
                                    lat_num=unraveling_hpc2hg_parameters['lat_num'])

fmap2hg2hpc = util.map_hg_to_hpc_rotate(fmap2hg, epi_lon=0.0, epi_lat=0.0, xbin=2.4, ybin=2.4)

# Difference should be zero everywhere in a perfect world!
#diff_util_hpc2hg2hpc = fmap.data - fmap2hg2hpc


# Test the difference between the wave simulation HG -> HPC transform and the
# util version of the HG -> HPC transform.



rmap2hpc = util.map_hg_to_hpc_rotate(rmap, epi_lon=epi_lon, epi_lat=epi_lat, xbin=2.4, ybin=2.4)

To look more closely we extract the code and put in on the top level

def map_hg_to_hpc_rotate(m, epi_lon=90, epi_lat=0, xbin=2.4, ybin=2.4):
Transform raw data in HG' coordinates to HPC coordinates

HG' = HG, except center at wave epicenter

m = deepcopy(fmap2hg) #rmap
xbin = 2.4 / unraveling_factor
ybin = 2.4 / unraveling_factor

# Origin grid, HG'
print("HG' information", [m.data.shape[1], m.data.shape[0]],
                                               [m.scale.x.value, m.scale.y.value],
                                               [m.reference_pixel.x.value, m.reference_pixel.y.value],
                                               [m.reference_coordinate.x.value, m.reference_coordinate.y.value])

lon_grid, lat_grid = wcs.convert_pixel_to_data([m.data.shape[1], m.data.shape[0]],
                                               [m.scale.x.value, m.scale.y.value],
                                               [m.reference_pixel.x.value, m.reference_pixel.y.value],
                                               [m.reference_coordinate.x.value, m.reference_coordinate.y.value])

# Origin grid, HG' to HCC'
# HCC' = HCC, except centered at wave epicenter
b0_deg = m.heliographic_latitude.to('degree').value
l0_deg = m.carrington_longitude.to('degree').value
print(b0_deg, l0_deg)
x, y, z = wcs.convert_hg_hcc(lon_grid, lat_grid,
                             b0_deg=b0_deg,
                             l0_deg=l0_deg,
                             z=True)

# Origin grid, HCC' to HCC''
# Moves the wave epicenter to initial conditions
# HCC'' = HCC, except assuming that HGLT_OBS = 0
zpp, xpp, ypp = util.euler_zyz((z, x, y),
                               (epi_lon, 90.-epi_lat, 0.))

# Origin grid, HCC to HPC (arcsec)
# xx, yy = wcs.convert_hcc_hpc(current_wave_map.header, xpp, ypp)
xx, yy = wcs.convert_hcc_hpc(xpp, ypp,
                             dsun_meters=m.dsun.to('meter').value)

# Destination HPC grid
hpcx_range = (np.nanmin(xx), np.nanmax(xx))
hpcy_range = (np.nanmin(yy), np.nanmax(yy))
print("HPC x y range", hpcx_range, hpcy_range)


hpcx = np.arange(hpcx_range[0], hpcx_range[1], xbin)
hpcy = np.arange(hpcy_range[0], hpcy_range[1], ybin)
newgrid_x, newgrid_y = np.meshgrid(hpcx, hpcy)

# Coordinate positions (HPC) with corresponding map data
points = np.vstack((xx.ravel(), yy.ravel())).T
values = np.array(deepcopy(m.data)).ravel()

# 2D interpolation from origin grid to destination grid
newdata = griddata(points[zpp.ravel() >= 0],
                   values[zpp.ravel() >= 0],
                   (newgrid_x, newgrid_y),
                   method="linear")

dict_header = {
    "CDELT1": xbin,
    "NAXIS1": len(hpcx),
    "CRVAL1": hpcx.min(),
    "CRPIX1": crpix12_value_for_HPC,
    "CUNIT1": "arcsec",
    "CTYPE1": "HPLN-TAN",
    "CDELT2": ybin,
    "NAXIS2": len(hpcy),
    "CRVAL2": hpcy.min(),
    "CRPIX2": crpix12_value_for_HPC,
    "CUNIT2": "arcsec",
    "CTYPE2": "HPLT-TAN",
    "HGLT_OBS": m.heliographic_latitude.to('degree').value,  # 0.0
    # "HGLN_OBS": 0.0,
    "CRLN_OBS": m.carrington_longitude.to('degree').value,  # 0.0
    'DATE_OBS': m.meta['date-obs'],
    'DSUN_OBS': m.dsun.to('m').value
}

rmap2hpc = Map(newdata, MapMeta(dict_header))


rmap2hpcs = rmap2hpc.superpixel((unraveling_factor, unraveling_factor)*u.pix, func=np.mean)


# Compare the two transforms. Should be identical.
diff_due2different_transforms = tmap.submap(rmap2hpc.xrange, rmap2hpc.yrange).data - rmap2hpcs.data

"""


