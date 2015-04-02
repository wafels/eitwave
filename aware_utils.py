#
# Utility functions for AWARE
#
import os
import util
import copy
import pickle

from datetime import timedelta, datetime
import numpy as np
from sunpy.net import helioviewer, vso
from sunpy.time import TimeRange, parse_time
from sunpy.wcs import convert_hpc_hg
from sunpy.map import Map
import sunpy.sun as sun
import astropy.units as u
from pb0r import pb0r
from sunpy.image.coalignment import repair_image_nonfinite

#
# Constants used in other parts of aware
#
solar_circumference_per_degree = 2 * np.pi * sun.constants.radius.to('m') / (360.0 * u.degree)
m2deg = 1.0 / solar_circumference_per_degree



def params(flare, **kwargs):
    """ Given a SunPy HEK flare object, extract the parameters required to
    transform maps.
    """
    if flare["event_coordunit"] == "degrees":
        flare_event_coord1 = flare['event_coord1']
        flare_event_coord2 = flare['event_coord2']
    elif flare["event_coordunit"] == "arcsec" or flare["event_coordunit"] == "arcseconds":
        info = pb0r(flare["event_starttime"])
        #Caution: the following conversion does not take dsun into account (i.e., apparent radius)
        flare_coords = convert_hpc_hg(flare['event_coord1'],
                                      flare['event_coord2'],
                                      info["b0"], info["l0"])
        flare_event_coord1 = flare_coords[0]
        flare_event_coord2 = flare_coords[1]

    """ Define the parameters we will use for the unraveling of the maps"""
    params = {"epi_lat": flare_event_coord2, #30., #degrees, HG latitude of wave epicenter
              "epi_lon": flare_event_coord1, #45., #degrees, HG longitude of wave epicenter
              #HG grid, probably would only want to change the bin sizes
              "lat_min": -90.,
              "lat_max": 90.,
              "lat_bin": 0.2,
              "lon_min": -180.,
              "lon_max": 180.,
              "lon_bin": 5.,
              #    #HPC grid, probably would only want to change the bin sizes
              "hpcx_min": -1025.,
              "hpcx_max": 1023.,
              "hpcx_bin": 2.,
              "hpcy_min": -1025.,
              "hpcy_max": 1023.,
              "hpcy_bin": 2.,
              "hglt_obs": 0,
              "rotation": 360. / (27. * 86400.), #degrees/s, rigid solar rotation
              }

    return params


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


def acquire_fits(directory, time_range, observatory='SDO', instrument='AIA',
                detector='AIA', measurement='211', verbose=True):
    """Acquire FITS files within the specified time range."""
    client=vso.VSOClient()
    tstart=time_range.t1.strftime('%Y/%m/%d %H:%M')
    tend=time_range.t2.strftime('%Y/%m/%d %H:%M')

    #check if any files are already in the directory
    current_files=[f for f in os.listdir(os.path.expanduser(directory)) if f.endswith('.fits')]
    
    #search VSO for FITS files within the time range, searching for AIA 211A only at a 36s cadence
    print 'Querying VSO to find FITS files...'
    qr=client.query(vso.attrs.Time(tstart,tend),vso.attrs.Instrument('aia'),vso.attrs.Wave(211,211),vso.attrs.Sample(36))

    dir=os.path.expanduser(directory)
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

        if not any(exists) == True:
            res=client.get([q],path=os.path.join(dir,'{file}.fits')).wait()
        else:
            print 'File at time ' + filetimestring + ' already exists. Skipping'

    fits_list=[os.path.join(dir,f) for f in os.listdir(dir) if f.endswith('.fits')]

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


def accumulate_from_file_list(filelist, accum=2, nsuper=4, normalize=True,
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
                map0 = (Map(filename)).superpixel((nsuper, nsuper))

            # Get the next map
            map1 = (Map(filename)).superpixel((nsuper, nsuper))

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


def map_unravel(mapcube, params, verbose=True):
    """ Unravel the maps in SunPy mapcube into a rectangular image. """
    new_maps = []
    for index, m in enumerate(mapcube):
        if verbose:
            print("Unraveling map %(#)i of %(n)i " % {'#': index + 1, 'n': len(mapcube)})
        unraveled = util.map_hpc_to_hg_rotate(m,
                                              epi_lon=params.get('epi_lon').to('degree').value,
                                              epi_lat=params.get('epi_lat').to('degree').value,
                                              lon_bin=params.get('lon_bin').to('degree').value,
                                              lat_bin=params.get('lat_bin').to('degree').value)
        # Should replace the NAN data with the local average of the non-nanned
        # data
        # new_map_data = repair_image_nonfinite(unraveled.data)
        new_maps.append(Map(unraveled.data, unraveled.meta))
    return Map(new_maps, cube=True)


def map_reravel(unravelled_maps, params, verbose=True):
    """ Transform rectangular maps back into heliocentric image. """
    reraveled_maps =[]
    for index, m in enumerate(unravelled_maps):
        if verbose:
            print("Transforming back to heliocentric coordinates map %(#)i of %(n)i " % {'#':index+1, 'n':len(unravelled_maps)})
        reraveled = util.map_hg_to_hpc_rotate(m,
                                        epi_lon=params.get('epi_lon').to('degree').value,
                                        epi_lat=params.get('epi_lat').to('degree').value,
                                        xbin=2.4,
                                        ybin=2.4)
        reraveled.data[np.isnan(reraveled)]=0.0
        reraveled_maps += [reraveled]
    return reraveled_maps


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
def score_long(nsector, isfinite, v, a, sigma_d, d, nt):
    # Velocity fit - implicit units are km/s
    kms = u.kilometer / u.second
    kms2 = u.kilometer / u.second / u.second
    print v
    print a
    if (v > 1.0 * kms) and (v < 2000.0 * kms):
        vscore = 1.0
    else:
        vscore = 0.0

    # Acceleration fit - implicit units are km/s/s
    if (a > -2.0 * kms2) and (a < 2.0 * kms2):
        ascore = 1.0
    else:
        ascore = 0.0

    # Distance fit
    gtz = d > 0.0
    sigma_rel = np.mean(sigma_d[gtz] / d[gtz])
    if sigma_rel < 0.5:
        sigma_rel_score = 1.0
    else:
        sigma_rel_score = 0.0

    # Final dynamic component of the score
    dynamic_component = (vscore + ascore + sigma_rel_score) / 6.0

    # Existence component
    existence_component = arc_duration_fraction(isfinite, nt) / 2.0

    print 'Dynamic component ', vscore, ' ', ascore, ' ', sigma_rel_score
    print 'Existence component ', existence_component
    print 'Total ', (existence_component + dynamic_component) * 100.0

    # Return the score in the range 0-100
    return existence_component + dynamic_component
