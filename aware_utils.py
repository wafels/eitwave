#
# Utility functions for AWARE
#
import numpy as np
import sunpy
import sunpy.map
import os
import util
import copy
from sunpy.net import helioviewer, vso
from sunpy.time import TimeRange, parse_time
from sunpy.wcs import convert_hpc_hg
from pb0r import pb0r
from datetime import timedelta, datetime


def params(flare, **kwargs):
    """ Given a SunPy HEK flare object, extract the parameters required to
    transform maps.
    """

    m2deg = 360.0 / (2 * 3.1415926 * 6.96e8)
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
            map1 = (sunpy.map.Map(filename)).superpixel((nsuper, nsuper))
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
        new_meta = copy.deepcopy(map1.meta)
        new_meta['exptime'] = np.float64(accum)
        maps.append(sunpy.map.Map(m , new_meta))
        if verbose:
            print('Accumulated map List has length %(#)i' % {'#': len(maps)})
    return maps


def map_unravel(mapcube, params, verbose=True):
    """ Unravel the maps in SunPy mapcube into a rectangular image. """
    new_maps = []
    for index, m in enumerate(mapcube.maps):
        if verbose:
            print("Unraveling map %(#)i of %(n)i " % {'#':index+1, 'n':len(mapcube.maps)})
        unraveled = util.map_hpc_to_hg_rotate(m,
                                               epi_lon=params.get('epi_lon'),
                                               epi_lat=params.get('epi_lat'),
                                               lon_bin=params.get('lon_bin'),
                                               lat_bin=params.get('lat_bin'))
            #print type(unraveled)
            #test=np.isnan(unraveled)
            #print len(test)
            #print test[0:10]
            #print unraveled.data[0:10]
        unraveled.data[np.isnan(unraveled)] = 0.0
        new_maps += [unraveled]
    return Map(new_maps, cube=True)


def map_reravel(unravelled_maps, params, verbose=True):
    """ Transform rectangular maps back into heliocentric image. """
    reraveled_maps =[]
    for index, m in enumerate(unravelled_maps):
        if verbose:
            print("Transforming back to heliocentric coordinates map %(#)i of %(n)i " % {'#':index+1, 'n':len(unravelled_maps)})
        reraveled = util.map_hg_to_hpc_rotate(m,
                                        epi_lon=params.get('epi_lon'),
                                        epi_lat=params.get('epi_lat'),
                                        xbin=2.4,
                                        ybin=2.4)
        reraveled.data[np.isnan(reraveled)]=0.0
        reraveled_maps += [reraveled]
    return reraveled_maps


def fit_wavefront(diffs, detection):
    """Fit the wavefront that has been detected by the hough transform.
    Simplest case is to fit along the y-direction for some x or range of x."""
    dims=diffs[0].shape
    answers=[]
    wavefront_maps=[]
    for i in range (0, len(diffs)):
        if (detection[i].max() == 0.0):
            #if the 'detection' array is empty then skip this image
            fit_map=sunpy.map.Map(np.zeros(dims),diffs[0].meta)
            print("Nothing detected in image " + str(i) + ". Skipping.")
            answers.append([])
            wavefront_maps.append(fit_map)
        else:
            #if the 'detection' array is not empty, then fit the wavefront in the image
            img = diffs[i]
            fit_map=np.zeros(dims)

            #get the independent variable for the columns in the image
            x=(np.linspace(0,dims[0],num=dims[0])*img.scale['y']) + img.yrange[0]
            
            #use 'detection' to guess the centroid of the Gaussian fit function
            guess_index=detection[i].argmax()
            guess_index=np.unravel_index(guess_index,detection[i].shape)
            guess_position=x[guess_index[0]]
            
            print("Analysing wavefront in image " + str(i))
            column_fits=[]
            #for each column in image, fit along the y-direction a function to find wave parameters
            for n in range (0,dims[1]):
                #guess the amplitude of the Gaussian fit from the difference image
                guess_amp=np.float(img.data[guess_index[0],n])
                
                #put the guess input parameters into a vector
                guess_params=[guess_amp,guess_position,5]

                #get the current image column
                y=img.data[:,n]
                y=y.flatten()                
                #call Albert's fitting function
                result = util.fitfunc(x,y,'Gaussian',guess_params)

                #define a Gaussian function. Messy - clean this up later
                gaussian = lambda p,x: p[0]/np.sqrt(2.*np.pi)/p[2]*np.exp(-((x-p[1])/p[2])**2/2.)
                
                #Draw the Gaussian fit for the current column and save it in fit_map
                #save the best-fit parameters in column_fits
                #only want to store the successful fits, discard the others.
                #result contains a pass/fail integer. Keep successes ( ==1).
                if result[1] == 1:
                    #if we got a pass integer, perform some other checks to eliminate unphysical values
                    result=check_fit(result)    
                    column_fits.append(result)
                    if result != []:
                        fit_column = gaussian(result[0],x)
                    else:
                        fit_column = np.zeros(len(x))
                else:
                    #if the fit failed then save as zeros/null values
                    result=[]
                    column_fits.append(result)
                    fit_column = np.zeros(len(x))
                
                #draw the Gaussian fit for the current column and save it in fit_map
                #gaussian = lambda p,x: p[0]/np.sqrt(2.*np.pi)/p[2]*np.exp(-((x-p[1])/p[2])**2/2.)
                    
                #save the drawn column in fit_map
                fit_map[:,n] = fit_column
            #save the fit parameters for the image in 'answers' and the drawn map in 'wavefront_maps'
            fit_map=sunpy.map.Map(fit_map,diffs[0].meta)
            answers.append(column_fits)
            wavefront_maps.append(fit_map)

    return answers, wavefront_maps

def wavefront_velocity(answers):
    """calculate wavefront velocity based on fit parameters for each column of an image or set of images"""
    velocity=[]
    for i in range(0,len(answers)):
        v=[]
        if i==0:
            velocity.append([])
        else:
            #skip blank entries of answers
            if answers[i] == [] or answers[i-1] == []:
                velocity.append([])
            else:
                for j in range(0,len(answers[i])):
                    #want to ignore null values for wave position
                    if answers[i][j] == [] or answers[i-1][j] == []:
                        vel=[]
                    else:             
                        vel=answers[i][j][0][1] - answers[i-1][j][0][1]
                    v.append(vel)
                velocity.append(v)
    return velocity


def wavefront_position_and_width(answers):
    """get wavefront position and width based on fit parameters for each column of an image or set of images"""
    position=[]
    width=[]
    for i in range(0,len(answers)):
        p=[]
        w=[]
        if answers[i] == []:
            position.append([])
            width.append([])
        else:
            for j in range(0,len(answers[i])):
                #want to ignore null values for wave position
                if answers[i][j] == []:
                    pos=[]
                    wid=[]
                else:
                    pos=answers[i][j][0][1]
                    wid=answers[i][j][0][2]
                p.append(pos)
                w.append(wid)
            position.append(p)
            width.append(w)
    return position,width

def fillLine(pos1,pos2,img):
    shape=img.shape
    ny = shape[0]
    nx = shape[1]
    if pos2[0] == pos1[0]:
        m = 9999
    else:
        m = (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
        
    constant = (pos2[1] - m*pos2[0])
    
    for x in range(pos1[0],pos2[0]):
        y = m*x + constant
        if y <= ny-1 and y>= 0:
            img[y,x] = 255

    return img

def htLine(distance,angle,img):
    shape = img.shape
    ny = shape[0]
    nx = shape[1]
    eps = 1.0/float(ny)

    if abs(np.sin(angle)) > eps:
        gradient = - np.cos(angle) / np.sin(angle)
        constant = distance / np.sin(angle)
        for x in range(0,nx):
            y = gradient*x + constant
            if y <= ny-1 and y >= 0:
                img[y,x] = 1
    else:
        img[:,distance] = 1

    return img

#
# Long et al (2014) score function
#
def score_long(nsector, isfinite, v, a, sigma_d, d):
    # Velocity fit - implicit units are km/s
    if (v > 1.0) and (v < 2000.0):
        vscore = 1.0
    else:
        vscore = 0.0

    # Acceleration fit - implicit units are m/s/s
    if (a > -2000.0) and (a < 2000.0):
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
    j = len(isfinite) - 1
    while isfinite[j] == False:
        j = j -1
    ntotal = j
    existence_component = nsector / np.float64(ntotal) / 2.0

    print 'Dynamic component ', vscore, ' ', ascore, ' ', sigma_rel_score
    print 'Existence component ', existence_component

    # Return the score in the range 0-100
    return (existence_component + dynamic_component) * 100.0
