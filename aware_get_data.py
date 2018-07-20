#
# Data acquisition functionality for AWARE
#
import os
import copy
import pickle
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from sunpy.net import helioviewer, vso, hek
from sunpy.time import TimeRange, parse_time
from sunpy.map import Map

import aware_constants

#
# Root of where the data is kept
#
eitwave_data_root = aware_constants.eitwave_data_root


###############################################################################
#
# Data download functions
#
# Download trigger events
#
def download_trigger_events(time_range,
                            event_type='FL',
                            frm='SSW Latest Events',
                            root_directory=eitwave_data_root):
    """
    Function to obtain potential wave triggering events from the HEK.
    """
    # Main directory holding the results
    subdirectory = os.path.join(root_directory, 'source', event_type)
    make_directory(subdirectory)

    # Make the query
    tr = TimeRange(time_range)
    hek_client = hek.HEKClient()
    result = hek_client.query(hek.attrs.Time(tr.start, tr.end),
                              hek.attrs.EventType(event_type),
                              hek.attrs.FRM.Name == frm)

    # Dump out the data
    pkl_file_name = 'hek.{:s}_{:s}.{:s}.{:s}.pkl'.format(str(tr.start), str(tr.end), event_type, frm)
    pkl_file = open(os.path.join(subdirectory, pkl_file_name), 'wb')
    pickle.dump(result, pkl_file)
    pkl_file.close()

    return result


#
# Download FITS files
#
def find_fits(time_range,
              instrument='AIA',
              measurement=211,
              sample=1*u.s):
    """
    Acquire FITS files within the specified time range.
    """
    client = vso.VSOClient()

    # Subdirectory where the FITS files will be stored
    instrument_measurement = os.path.join(instrument, str(measurement))

    # Search VSO for FITS files within the time range
    tr = TimeRange(time_range)
    return instrument_measurement, client.query(vso.attrs.Time(tr.start, tr.end),
                                                vso.attrs.Instrument(instrument),
                                                vso.attrs.Wave(measurement*u.AA, measurement*u.AA),
                                                vso.attrs.Sample(sample))


def download_fits(qr, instrument_measurement=None,
                  root_directory=eitwave_data_root,
                  extension='fits'):
    """
    Acquire FITS files within the specified time range.
    Here is an example script to download directly from the JSOC using SunPy 0.9+.
    Note that you need to wait until the JSOC has built your request before
    attempting a download.

    With SunPy 0.9+
    ---------------
    from sunpy.net import Fido, attrs as a
    import astropy.units as u
    qr = Fido.search(a.jsoc.Time('2011-02-15 01:48:00', '2011-02-15 02:48:00'), a.jsoc.Series('aia.lev1_euv_12s'), a.Sample(12*u.s), a.jsoc.Notify('email@email.org'), a.jsoc.Wavelength(211*u.AA))
    downloaded_files = Fido.fetch(qr)

    With SunPy 0.7.10
    -----------------
    import astropy.units as u
    from sunpy.net import jsoc
    client = jsoc.JSOCClient()
    response = client.query(jsoc.Time('2014-01-01T00:00:00', '2014-01-01T01:00:00'), jsoc.Segment('image'), jsoc.Series('aia.lev1_euv_12s'), jsoc.Notify("sunpy@sunpy.org"), jsoc.Wavelength(211*u.AA))
    res = client.get(response)
    """
    client = vso.VSOClient()

    # Subdirectory where the FITS files will be stored
    subdirectory = os.path.join(root_directory, instrument_measurement, extension, '1.0')
    make_directory(subdirectory)

    for q in qr:
        res = client.get([q], path=os.path.join(subdirectory, '{file}.fits')).wait()

    # List of files
    return get_file_list(subdirectory, extension)


###############################################################################
#
# Data manipulation
#
# Create a mapcube from a list of files.  In order to save memory, this
# function also sums in the temporal and spatial directions.
#
def accumulate_from_file_list(file_list,
                              temporal_summing=2,
                              spatial_summing=[4, 4]*u.pix,
                              normalize=True,
                              verbose=False):
    """
    Add up data in time and space. Accumulate 'temporal_summing' files in time,
    and then form the images into spatial_summing super pixels.  Returns the
    sum of all the exposure rates.  Sending in a file list and accumulating it
    is cheaper in terms of memory as opposed to reading all the files in to a
    large datacube and then performing the accumulation step.
    """
    # counter for number of files.
    j = 0

    # storage for the returned maps
    maps = []

    # Sum in time and in space
    n_files = len(file_list)
    while j + temporal_summing <= n_files:
        i = 0
        while i < temporal_summing:
            filename = file_list[i + j]
            if verbose:
                print('File %(#)i out of %(nfiles)i' % {'#': i + j, 'n_files': n_files})
                print('Reading in file ' + filename)
            # Get the initial map
            if i == 0:
                map0 = (Map(filename)).superpixel(spatial_summing)
                print(i, map0.date)

            # Get the next map
            map1 = (Map(filename)).superpixel(spatial_summing)
            print(i, map1.date)

            # Normalization
            if normalize:
                normalization = map1.exposure_time
            else:
                normalization = 1.0

            if i == 0:
                # Emission rate
                m = map1.data / normalization
            else:
                # Emission rate
                m += map1.data / normalization
            i += 1
        j += temporal_summing

        # Make a copy of the meta header and set the exposure time to
        # temporal_summing, indicating that 'n' normalized exposures were used.
        new_meta = copy.deepcopy(map0.meta)
        new_meta['exptime'] = np.float64(temporal_summing)
        maps.append(Map(m, new_meta))
        if verbose:
            print('Accumulated map List has length %(#)i' % {'#': len(maps)})
    return Map(maps, cube=True)


###############################################################################

def make_directory(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def get_file_list(directory, extension):
    """ get the file list and sort it.  For well behaved file names the file
    name list is returned ordered by time."""
    lst = []
    loc = os.path.expanduser(directory)
    for f in os.listdir(loc):
        if f.endswith(extension):
            lst.append(os.path.join(loc, f))
    return sorted(lst)


###############################################################################
#
# More general purpose data download
#
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
        data = acquire_fits(directory, data_range)
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