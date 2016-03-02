#
# Utility functions for AWARE
#
import os
import copy
import pickle
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from sunpy.net import helioviewer, vso
from sunpy.time import TimeRange, parse_time
from sunpy.map import Map
import sunpy.sun as sun

#
# Constants used in other parts of aware
#
solar_circumference_per_degree = 2 * np.pi * sun.constants.radius.to('m') / (360.0 * u.degree)
m2deg = 1.0 / solar_circumference_per_degree
solar_circumference_per_degree_in_km = solar_circumference_per_degree.to('km/deg') * u.degree


def dump_images(mc, directory, name):
    for im, m in enumerate(mc):
        fname = '%s_%05d.png' % (name, im)
        dump_image(m.data, directory, fname)


def dump_image(img, directory, name):
    ndir = os.path.expanduser('~/eitwave/img/%s/' % directory)
    if not(os.path.exists(ndir)):
        os.makedirs(ndir)
    plt.ioff()
    plt.imshow(img)
    plt.savefig(os.path.join(ndir, name))


def acquire_data(directory, extension, flare, duration=60, verbose=True):

    # vals = eitwaveutils.goescls2number( [hek['fl_goescls'] for hek in
    # hek_result] )
    # flare_strength_index = sorted(range(len(vals)), key=vals.__getitem__)
    # Get the data for each flare.
    if verbose:
        print('Event start time: ' + flare['event_starttime'])
        print('GOES Class: ' + flare['fl_goescls'])
    data_range = TimeRange(parse_time(flare['event_starttime']),
                           parse_time(flare['event_starttime']) +
                           timedelta(minutes=duration))
    if extension.lower() == '.jp2':
        data = acquire_jp2(directory, data_range)
    if extension.lower() in ('.fits', '.fts'):
        data = acquire_fits(directory,data_range)
    # Return the flare list from the HEK and a list of files for each flare in
    # the HEK flare list
    return data


def listdir_fullpath(d, filetype=None):
    dd = os.path.expanduser(d)
    filelist = os.listdir(dd)
    if filetype == None:
        return sorted([os.path.join(dd, f) for f in filelist])
    else:
        filtered_list = []
        for f in filelist:
            if f.endswith(filetype):
                filtered_list.append(f)
        return sorted([os.path.join(dd, f) for f in filtered_list])


def get_jp2_dict(directory):
    directory_listing = {}
    l = sorted(os.listdir(os.path.expanduser(directory)))
    for f in l:
        dt = hv_filename2datetime(f)
        directory_listing[dt] = os.path.join(os.path.expanduser(directory), f)
    return directory_listing


def hv_filename2datetime(f):
    try:
        ymd = f.split('__')[0]
        hmsbit = f.split('__')[1]
        hms = hmsbit.split('_')[0] + '_' + hmsbit.split('_')[1] + '_' + \
            hmsbit.split('_')[2]
        dt = datetime.strptime(ymd + '__' + hms, '%Y_%m_%d__%H_%M_%S')
    except:
        dt = None
    return dt


def acquire_jp2(directory, time_range, observatory='SDO', instrument='AIA',
                detector='AIA', measurement='211', verbose=True):
    """Acquire Helioviewer JPEG2000 files between the two specified times"""

    # Create a Helioviewer Client
    hv = helioviewer.HelioviewerClient()

    # Start the search
    jp2_list = []
    this_time = time_range.t1
    while this_time <= time_range.t2:
        # update the directory dictionary with the latest contents
        directory_dict = get_jp2_dict(directory)

        # find what the closest image to the requested time is
        response = hv.get_closest_image(this_time, observatory=observatory,
                              instrument=instrument, detector=detector,
                              measurement=measurement)

        # if this date is not already present, download it
        if not(response["date"] in directory_dict):
            if verbose:
                print('Downloading new file:')
            jp2 = hv.download_jp2(this_time, observatory=observatory,
                              instrument=instrument, detector=detector,
                              measurement=measurement, directory=directory,
                              overwrite=True)
        else:
            # Otherwise, get its location
            jp2 = directory_dict[response["date"]]
        # Only one instance of this file should exist
        if not(jp2 in jp2_list):
            jp2_list.append(jp2)
            if verbose:
                print('Found file ' + jp2 + '. Total found: ' + str(len(jp2_list)))

        # advance the time
        this_time = this_time + timedelta(seconds=6)
    return jp2_list


#
# Data acquisition functions
#
def acquire_fits(directory, time_range, instrument='AIA',
                detector='AIA', measurement=211*u.AA, verbose=True):
    """Acquire FITS files within the specified time range."""
    client = vso.VSOClient()
    tstart = time_range.t1.strftime('%Y/%m/%d %H:%M')
    tend = time_range.t2.strftime('%Y/%m/%d %H:%M')

    # check if any files are already in the directory
    current_files = [f for f in os.listdir(os.path.expanduser(directory)) if f.endswith('.fits')]
    
    # Search VSO for FITS files within the time range,
    # searching for AIA 211A only at a 36s cadence
    print 'Querying VSO to find FITS files...'
    time = vso.attrs.Time(tstart, tend)
    instrument = vso.attrs.Instrument(instrument)
    wavelength = vso.attrs.Wavelength(, 211*u.AA)
    qr = client.query(time, instrument, wavelength)

    dir = os.path.expanduser(directory)
    print 'Downloading '+str(len(qr))+ ' files from VSO to ' + dir

    for q in qr:
        filetimestring=q.time.start[0:4] + '_' + q.time.start[4:6] + '_' + q.time.start[6:8] + 't' \
          + q.time.start[8:10] + '_' +q.time.start[10:12] + '_' + q.time.start[12:14]

        exists=[]
        for c in current_files:
            if filetimestring in c:
                exists.append(True)
            else:
                exists.append(False)

        if not any(exists):
            res = client.get([q],path=os.path.join(dir,'{file}.fits')).wait()
        else:
            print 'File at time ' + filetimestring + ' already exists. Skipping'

    fits_list=[os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.fits')]

    return fits_list


def get_file_list(directory, extension):
    """ get the file list and sort it.  For well behaved file names the file
    name list is returned ordered by time"""
    lst = []
    loc = os.path.expanduser(directory)
    for f in os.listdir(loc):
        if f.endswith(extension):
            lst.append(os.path.join(loc, f))
    return sorted(lst)


def accumulate_from_file_list(filelist, accum=2, nsuper=[4, 4]*u.pix,
                              normalize=True,
                              verbose=False):
    """
    Add up data in time and space. Accumulate 'accum' files in time, and
    then form the images into super by super superpixels.  Returns the
    sum of all the exposure rates.  Sending in a file list and accumulating it
    is cheaper in terms of memory as opposed to reading all the files in to a
    large datacube and then performing the accumulation step.
    """
    # counter for number of files.
    j = 0
    # storage for the returned maps
    maps = []
    nfiles = len(filelist)
    while j + accum <= nfiles:
        i = 0
        while i < accum:
            filename = filelist[i + j]
            if verbose:
                print('File %(#)i out of %(nfiles)i' % {'#': i + j, 'nfiles':nfiles})
                print('Reading in file ' + filename)
            # Get the initial map
            if i == 0:
                map0 = (Map(filename)).superpixel(nsuper)

            # Get the next map
            map1 = (Map(filename)).superpixel(nsuper)

            # Normalizaion
            if normalize:
                normalization = map1.exposure_time
            else:
                normalization = 1.0

            if i == 0:
                # Emission rate
                m = map1.data / normalization
            else:
                # Emission rate
                m = m + map1.data / normalization
            i = i + 1
        j = j + accum
        # Make a copy of the meta header and set the exposure time to accum,
        # indicating that 'n' normalized exposures were used.
        new_meta = copy.deepcopy(map0.meta)
        new_meta['exptime'] = np.float64(accum)
        maps.append(Map(m , new_meta))
        if verbose:
            print('Accumulated map List has length %(#)i' % {'#': len(maps)})
    return maps


def write_movie(mc, filename, start=0, end=None):
    """
    Write a movie standard movie out from the input datacube

    Parameters
    ----------
    :param mc: input mapcube
    :param filename: name of the movie file
    :param start: first element in the mapcube
    :param end: last element in the mapcube
    :return: output_filename: filename of the movie.
    """
    FFMpegWriter = animation.writers['ffmpeg']
    fig = plt.figure()
    metadata = dict(title=name, artist='Matplotlib', comment='AWARE')
    writer = FFMpegWriter(fps=15, metadata=metadata, bitrate=2000.0)
    output_filename = filename + '.mp4'
    with writer.saving(fig, output_filename, 100):
        for i in range(start, len(mc)):
            mc[i].plot()
            mc[i].draw_limb()
            mc[i].draw_grid()
            plt.title(mc[i].date)
            writer.grab_frame()
    return output_filename


def get_trigger_events(eventname):
    """
    Function to obtain potential wave triggering events from the HEK.
    """
    # Main directory holding the results
    pickleloc = aware_utils.storage(eventname)
    # The filename that stores the triggering event
    hek_trigger_filename = aware_utils.storage(eventname, hek=True)
    pkl_file_location = os.path.join(pickleloc, hek_trigger_name)

    if not os.path.exists(pickleloc):
        os.makedirs(pickleloc)
        hclient = hek.HEKClient()
        tr = info[eventname]["tr"]
        ev = hek.attrs.EventType('FL')
        result = hclient.query(tr, ev, hek.attrs.FRM.Name == 'SSW Latest Events')
        pkl_file = open(pkl_file_location, 'wb')
        pickle.dump(result, pkl_file)
        pkl_file.close()
    else:
        pkl_file = open(pkl_file_location, 'rb')
        result = pickle.load(pkl_file)
        pkl_file.close()


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
