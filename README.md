1. AWARE - Automated Wave Analysis and REduction
------------------------------------------------
This is experimental code designed to detect and characterize EUV waves in EUV image data.

2.  Basic structure of how to run AWARE
---------------------------------------
The program to detect wave that you want to run is 

*swave_characterize2.py*

which obtains, loads and manipulates the data to measure waves.  It is controlled in large part by

*swave_study.py*

which controls all the various aspects of AWARE.  We have tried to document every feature that can be tweaked, but if anything is unclear, please file an issue.

After setting up how I want to run AWARE using *swave_study.py*, it gets run in an IPython session as *%run swave_characterize2.py*.


Observational Characteristics
-----------------------------

These are the observational characteristics of EUV waves as reported in literature

from [Warmuth et al. (2010)](http://adsabs.harvard.edu/abs/2010AdSpR..45..527W)

 * "We stress that all studied events show the same basic characteristics: the signatures in all wavelength ranges – Ha, HeI, EUV, and SXR – are consistent with a single underlying physical disturbance. In all cases, the disturbance is decelerating. This is seen from the combined kinematical curves as well as in the individual data sets from the various spectral channels. 
 * "There is a relation between the amount of deceleration and propagation speeds: faster waves tend to decelerate more strongly."

from [Warmuth et al. (2004)](http://adsabs.harvard.edu/abs/2004A%26A...418.1117W)

 * Fitting the profiles using a second-order polynomial fit (r = r0 + v0*t + 1/2*a0*t^2) for 12 events.
 * The mean parameters were found to be v0 = 933 +/- 252 km/s, a0 = −1495 +/- 1262 m/s^2.

from [Long et al. 2011](http://adsabs.harvard.edu/abs/2011SPD....42.0505L)

 * "Significant pulse broadening was also measured using both STEREO ( 55 km/s) andSDO ( 65 km/s) observations."

Download AGU 2011 test data from:
  [http://www.sunpy.org/research/agu-2011/testdata1.zip](http://www.sunpy.org/research/agu-2011/testdata1.zip)

Download SPD 2012 test data from:
  [http://umbra.nascom.nasa.gov/people/ireland/Data/eitwave/jp2/20110601_02_04.tar.gz](http://umbra.nascom.nasa.gov/people/ireland/Data/eitwave/jp2/20110601_02_04.tar.gz)


References
----------
 * [Hough Transform](http://en.wikipedia.org/wiki/Hough_transform)
 * [Mendelev list of papers](http://www.mendeley.com/groups/1335103/sunpy-eit-wave/papers/)
 * [The Kinematics and Energetics of Globally-propagating Coronal Disturbance](http://www.maths.tcd.ie/~dlong/Presentations/BUKS_apr_09.pdf)
