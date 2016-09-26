#
# Utility functions for AWARE
#
import os
import pickle
import numpy as np
import astropy.units as u
from sunpy.time import TimeRange, parse_time
from sunpy.map import Map
import aware_get_data
import aware_constants
import matplotlib.animation as animation
import matplotlib.pyplot as plt

eitwave_data_root = aware_constants.eitwave_data_root


def create_input_to_aware_for_test_observational_data(wave_name,

                                                      instrument='AIA',
                                                      wavelength=211,
                                                      event_type='FL',
                                                      root_directory=eitwave_data_root):
    # Set up the data
    if wave_name == 'longetal2014_figure4':
        hek_record_index = 0

    if wave_name == 'longetal2014_figure7':
        hek_record_index = 0

    if wave_name == 'longetal2014_figure8a':
        hek_record_index = 0
        time_range = ['2011-02-15 01:48:00', '2011-02-15 02:14:34']

    if wave_name == 'longetal2014_figure8e':
        hek_record_index = 0
        time_range = ['2011-02-16 14:22:36', '2011-02-16 14:39:48']

    if wave_name == 'byrneetal2013_figure12':
        hek_record_index = 0
        time_range = ['2010-08-14 09:40:18', '2010-08-14 10:32:00']

    # Where the data is stored
    wave_info_location = os.path.join(root_directory, wave_name)

    # Get the FITS files we are interested in
    fits_location = os.path.join(wave_info_location, instrument, str(wavelength), 'fits', '1.0')
    fits_file_list = aware_get_data.get_file_list(fits_location, '.fits')
    if len(fits_file_list) == 0:
        instrument_measurement, qr = aware_get_data.find_fits(time_range, instrument, wavelength)
        print('Downloading {:n} files'.format(len(qr)))
        fits_file_list = aware_get_data.download_fits(qr, instrument_measurement=instrument_measurement)

    # Get the source information
    source_location = os.path.join(wave_info_location, event_type)
    source_path = aware_get_data.get_file_list(source_location, '.pkl')
    if len(source_path) == 0:
        print('Querying HEK for trigger data.')
        hek_record = aware_get_data.download_trigger_events(time_range)
    else:
        f = open(source_path[0], 'rb')
        hek_record = pickle.load(f)
        f.close()

    analysis_time_range = TimeRange(hek_record[hek_record_index]['event_starttime'],
                                    time_from_file_name(fits_file_list[-1].split(os.path.sep)[-1]))

    for_analysis = []
    for f in fits_file_list:
        g = f.split(os.path.sep)[-1]
        if (time_from_file_name(g) <= analysis_time_range.end) and (time_from_file_name(g) >= analysis_time_range.start):
            for_analysis.append(f)

    return {'finalmaps': aware_get_data.accumulate_from_file_list(for_analysis),
            'epi_lat': hek_record[hek_record_index]['hgs_y'],
            'epi_lon': hek_record[hek_record_index]['hgs_x']}


def time_from_file_name(f, fits_level=1.0):
    if fits_level == 1.0:
        start = 14

    year = f[start:start+4]
    month = f[start+5:start+7]
    day = f[start+8:start+10]
    hour = f[start+11:start+13]
    minute = f[start+14:start+16]
    second = f[start+17:start+19]

    time = '{:s}-{:s}-{:s} {:s}:{:s}:{:s}'.format(year, month, day, hour, minute, second)

    return parse_time(time)


###############################################################################
#
# AWARE arc and wave measurement scores
#
def arc_duration_fraction(defined, nt):
    """
    :param defined: boolean array of where there is a detection
    :param nt: number of possible detections
    :return: fraction of the full duration that the detection exists
    """
    return np.float64(np.sum(defined)) / np.float64(nt)


#
# Long et al (2014) score function
#
class ScoreLong:
    """
    Calculate the Long et al (2014) score function.
    """
    def __init__(self, velocity, acceleration, sigma_d, d, nt,
                 velocity_range=[1, 2000] * u.km/u.s,
                 acceleration_range=[-2.0, 2.0] * u.km/u.s/u.s,
                 sigma_rel_limit=0.5):
        self.velocity = velocity
        self.acceleration = acceleration
        self.sigma_d = sigma_d
        self.d = d
        self.nt = nt
        self.velocity_range = velocity_range
        self.acceleration_range = acceleration_range
        self.sigma_rel_limit = sigma_rel_limit

        # Velocity fit - is it acceptable?
        if (self.velocity > self.velocity_range[0]) and (self.velocity < self.velocity_range[1]):
            self.velocity_score = 1.0
        else:
            self.velocity_score = 0.0
        self.velocity_is_dynamic_component = 1.0

        # Acceleration fit - is it acceptable?
        if self.acceleration is not None:
            self.acceleration_is_dynamic_component = 1.0
            if (self.acceleration > self.acceleration_range[0]) and (self.acceleration < self.acceleration_range[1]):
                self.acceleration_score = 1.0
            else:
                self.acceleration_score = 0.0
        else:
            self.acceleration_is_dynamic_component = 0.0
            self.acceleration_score = 0.0

        # Did the fit along the arc have a reasonable errors on average?
        self.sigma_rel = np.mean(sigma_d/d)
        if self.sigma_rel < self.sigma_rel_limit:
            self.sigma_rel_score = 1.0
        else:
            self.sigma_rel_score = 0.0
        self.sigma_is_dynamic_component = 1.0

        # Final dynamic component of the score
        self.n_dynamic_components = self.velocity_is_dynamic_component + \
                                    self.acceleration_is_dynamic_component + \
                                    self.sigma_is_dynamic_component
        self.dynamic_component = 0.5*(self.velocity_score +
                                      self.acceleration_score +
                                      self.sigma_rel_score) / self.n_dynamic_components

        # Existence component - how much of the data along the arc was fit?
        self.existence_component = 0.5 * len(self.d) / (1.0 * self.nt)

        # Return the score in the range 0-100
        self.final_score = 100*(self.existence_component + self.dynamic_component)


###############################################################################
#
# AWARE - make a plot of the progress of the detected wave front.
#
def progress_map(mc, index=0):
    """
    Take an input mapcube and return the detected progress of the wavefront as
    a single sunpy map.

    mc : sunpy.map.MapCube
        Input mapcube

    index : int
        Index of the map in the input mapcube that

    Return
    ------
    map, list

    A tuple containing the following: a sunpy map with values in the range 1 to
    len(input mapcube) where larger numbers indicate later times, and a list
    that holds the corresponding timestamps.  If a pixel in the map has value
    'n', then the wavefront is located at time timestamps[n].
    """

    wave_progress_data = np.zeros_like(mc[index].data)
    timestamps = []
    for im in range(0, len(mc)-1):
        detection1 = mc[im+1].data
        detection1[detection1 > 0.0] = 1
        detection0 = mc[im].data
        detection0[detection0 > 0.0] = 1
        progress_index = detection1 - detection0 > 0.0
        wave_progress_data[progress_index] = im + 1

        # Keep a record of the timestamps
        timestamps.append(mc[im+1].date)

    return Map(wave_progress_data, mc[index].meta), timestamps


###############################################################################
#
# AWARE - make a plot of the progress of the detected wave front.
#
def progress_mask(mc, lower_limit=0.0):
    """
    Take an input AWARE-processed detection and return a binary mask
    mapcube that shows where the data is.

    mc : sunpy.map.MapCube
        Input mapcube

    Return
    ------
    mapcube
    """

    pm = []
    for im, m in enumerate(mc):
        wave_location_mask = 1.0*(m.data > lower_limit)
        pm.append(Map(wave_location_mask, m.meta))

    return Map(pm, cube=True)


###############################################################################
#
# AWARE - make movie.
#

def draw_limb(fig, ax, sunpy_map):
    p = sunpy_map.draw_limb()
    return p


#
def write_movie(mc, filename):
    """
    Take a mapcube and produce a movie of it.

    :param mc:
    :param filename:
    :return:
    """
    ani = mc.plot(plot_function=draw_limb)
    Writer = animation.writers['avconv']
    writer = Writer(fps=10, metadata=dict(artist='SunPy'), bitrate=18000)
    ani.save('{:s}.mp4'.format(filename), writer=writer)
    plt.close('all')
