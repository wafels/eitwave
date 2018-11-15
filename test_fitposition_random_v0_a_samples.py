"""
A number of fit trials are run.  Each trial has a random initial
velocity and acceleration.  The random distribution used is the
uniform distribution within a specified range.  Plots are made illustrating
the
"""

import os
from copy import deepcopy
import matplotlib
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
axis_fontsize = 20
legend_fontsize_fraction = 0.75
text_fontsize_fraction = 0.75
matplotlib.rcParams.update({'font.size': axis_fontsize})

import re
import numpy as np
from numpy.random import uniform, randint
from scipy.stats import spearmanr
from scipy.stats.mstats import spearmanr as masked_spearmanr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['text.usetex'] = True
import astropy.units as u
from statsmodels.robust import mad
from aware5_without_swapping_emission_axis import FitPosition
from aware_constants import solar_circumference_per_degree

# Where to save the data.
image_directory = os.path.expanduser('~/eitwave/img/test_fitposition')
pad_inches = 0.05

# Save to file
save = True

# Show statistic used
show_statistic = False

# Type of statistic
use_median = True

# Maintain the overall duration of the time series?
ts = {"maintain": True, "accum": 3, "dt": 12*u.s, "nt": 60}

# Calculate the sample properties of the time series
if ts["maintain"]:
    nt = ts["nt"] // ts["accum"]
    dt = ts["accum"]*ts["dt"]
else:
    nt = ts["nt"]
    dt = ts["dt"]

# Noise level
sigma = 5*u.degree

# Initial displacement
s0 = 0*u.degree

# Initial velocity range
v0 = 1*u.km/u.s

# True accelerations to use
a0 = 1.0 * u.km/u.s/u.s

# Number of trials at each value of the acceleration
ntrial = 200

#
# Mean - or median - velocity and acceleration plots
#
a_true = r'$a_{\mbox{true}}$'
a_fit = r'$a_{\mbox{fit}}$'

v_true = r'$v_{\mbox{true}}$'
v_fit = r'$v_{\mbox{fit}}$'

v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')

# Storage for the results
sz1v = np.zeros(ntrial)
sz1ve = np.zeros_like(sz1v)
sz1b = np.zeros_like(sz1v)
sz2v = np.zeros_like(sz1v)
sz2ve = np.zeros_like(sz1v)
sz2a = np.zeros_like(sz1v)
sz2ae = np.zeros_like(sz1v)
sz2b = np.zeros_like(sz1v)
sz2f = np.zeros_like(sz1v, dtype=np.bool)

# Time range
t = dt*np.arange(0, nt)

vr = [0, 1000]
ar = [-1, 1]

# Go through all the trials
for i in range(0, ntrial):
    #
    nt_random = randint(low=10, high=60)
    t = dt * np.arange(0, nt_random)

    # Estimated error
    position_error = sigma*np.ones(nt_random)

    # Random initial velocity
    v0_random = v0 * uniform(low=vr[0], high=vr[1])
    v = (v0_random/solar_circumference_per_degree).to(u.deg/u.s)

    # random acceleration
    a0_random = a0 * uniform(low=ar[0], high=ar[1])
    a = (a0_random/solar_circumference_per_degree).to(u.deg/u.s/u.s)

    # Position of the wave
    position = s0 + v*t + 0.5*a*t*t

    # Noise
    noise = sigma*np.random.normal(loc=0.0, scale=1.0, size=nt_random)

    # Do the quadratic fit and store the values
    z2 = FitPosition(t, position + noise, position_error, n_degree=2)
    sz2v[i] = z2.velocity.value
    sz2ve[i] = z2.velocity_error.value
    sz2a[i] = z2.acceleration.value
    sz2ae[i] = z2.acceleration_error.value
    sz2b[i] = z2.BIC
    sz2f[i] = z2.fitted

    # Do the linear fit and store the values
    z1 = FitPosition(t, position + noise, position_error, n_degree=1)
    sz1v[i] = z1.velocity.value
    sz1ve[i] = z1.velocity_error.value
    sz1b[i] = z1.BIC

z1v = (sz1v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
z1ve = (sz1ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

z2v = (sz2v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
z2ve = (sz2ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

z2a = (sz2a * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value
z2ae = (sz2ae * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value

filename = os.path.expanduser('~/eitwave/dat/test_fitposition/test_fitposition_random_v0_and_a.npz')
np.savez(filename, z1v, z1ve, z2v, z2ve, z2a, z2ae, sz1b, sz2b)

#
# Create a results density plot of the acceleration and velocity fits
#
rho_spearman = '$\\rho_{s}$'

limit_types = ['simple']  # , 'Long et al')

answer = dict()
for i in limit_types:
    fitted = deepcopy(sz2f)
    if i == 'simple':
        # Simple physical limits applied
        fitted[z2v < 0] = False
    if i == 'Long et al':
        # Long et al limits applied
        fitted[z2v < 0] = False
        fitted[z2v > 2000] = False
        fitted[z2a < -2.0] = False
        fitted[z2a > 2.0] = False

    xx = np.ma.array(z2a, mask=~fitted)
    yy = np.ma.array(z2v, mask=~fitted)

    ccm = masked_spearmanr(xx, yy)
    this_poly = np.ma.polyfit(xx, yy, 1)

    cc_string = '{:s}={:.2f} (p={:.2f})'.format(rho_spearman, ccm.correlation, np.float(ccm.pvalue.data))

    answer[i] = {'xx': xx, 'yy': yy, 'ccm': ccm, "this_poly": this_poly, "cc_string": cc_string,
                 'z2ae': z2ae, 'z2ve': z2ve}

plt.ion()
fig, ax = plt.subplots()
# hist2d = ax.hist2d(z2a, z2v, bins=[40, 40])
for i in limit_types:
    xx = answer[i]['xx']
    yy = answer[i]['yy']
    z2ae = answer[i]['z2ae']
    z2ve = answer[i]['z2ve']
    ax.errorbar(xx, yy, xerr=z2ae, yerr=z2ve, linestyle='none')
label_fit = '{:s}={:.0f}{:s} + {:.0f}'.format(v_fit, this_poly[0], a_fit, this_poly[1])

xlim = ax.get_xlim()
a_x = np.linspace(xlim[0], xlim[1], 100)
for i in limit_types:
    best_fit = np.polyval(answer[i]["this_poly"], a_x)
    cc_string = answer[i]["cc_string"]
    ax.plot(a_x, best_fit, label='best fit ({:s})\n{:s}'.format(label_fit, cc_string))

ax.axhline(vr[0], linestyle=':', label='true initial velocity limits', color='k')
ax.axhline(vr[1], linestyle=':', color='k')
ax.axvline(ar[0], linestyle='-.', label='true initial acceleration limits', color='k')
ax.axvline(ar[1], linestyle='-.', color='k')
ax.set_xlabel('{:s} ({:s})'.format(a_fit, a_string))
ax.set_ylabel('{:s} ({:s})'.format(v_fit, v_string))
ax.set_title('(e) multiple varying arc length,\n {:s} and {:s}'.format(v_true, a_true))
ax.grid(linestyle=":")
ax.set_xlim(xlim)

plt.legend(framealpha=0.6, fontsize=12, facecolor='yellow')
plt.tight_layout()
if save:
    filename = 'test_fit_position_random_v0_a_samples.pdf'
plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)
