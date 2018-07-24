"""
Makes plots illustrating the bias in fitting.
This program creates plots for the paper illustrating
the fit bias by fitting noisy realizations of the same
true data many times.  It plots two-dimensional histograms
of the fitted velocity and fitted acceleration. Three
accelerations are considered - one positive, one
negative, and no acceleration
"""

import os
import matplotlib
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
matplotlib.rcParams.update({'font.size': 18})

this_file = 'test_fitbias_fitted_v_versus_a'

# Where to save the data.
image_directory = os.path.expanduser('~/eitwave/img/{:s}'.format(this_file))

image_root_name = this_file

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.rcParams['text.usetex'] = True
import astropy.units as u
from statsmodels.robust import mad
from aware5_without_swapping_emission_axis import FitPosition
from aware_constants import solar_circumference_per_degree

# Save to file
save = True

# Show statistic used
show_statistic = False

# Type of statistic
use_median = False

# Maintain the overall duration of the time series?
ts = {"maintain": True, "accum": 3, "dt": 12*u.s, "nt": 30}

# Calculate the sample properties of the time series
if ts["maintain"]:
    nt = ts["nt"] // ts["accum"]
    dt = ts["accum"]*ts["dt"]
else:
    nt = ts["nt"]
    dt = ts["dt"]

plot_labels = ['(a)', '(b)', '(c)']

# Noise level
sigma = 5*u.degree

# Initial displacement
s0 = 0*u.degree

# Initial velocity
speed_unit = u.km/u.s
v0 = 250 * speed_unit
v_true = r'$v_{\mbox{true}}$'
v_fit = r'$v_{\mbox{fit}}$'

# Estimated error
position_error = sigma*np.ones(nt)

# Plot range
a_x = np.linspace(-2, 2, 100)
v_y = np.linspace(-1500, 1500, 100)


# Three accelerations to consider
acceleration_unit = u.km/u.s/u.s
na = 3
a0 = 1.0 * acceleration_unit
a_true = r'$a_{\mbox{true}}$'
a_fit = r'$a_{\mbox{fit}}$'

# Number of trials at each value of the acceleration
ntrial = 10000

# Set up some plotting information
pad_inches = 0.05
sigma_string = '$\sigma=${:n}{:s}'.format(sigma.value, sigma.unit.to_string('latex_inline'))
sample_string = '$n_{t}=$'
trial_string = '{:s}{:n}, $\delta t=${:n}{:s}, {:n} trials'.format(sample_string, nt, dt.value, dt.unit.to_string('latex_inline'), ntrial)
subtitle = '\n{:s}, {:s}'.format(sigma_string, trial_string)

# Plotting help
v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')

# Storage for the results
fit_1_v = np.zeros((na, ntrial))
fit_1_ve = np.zeros_like(fit_1_v)
fit_1_b = np.zeros_like(fit_1_v)
fit_1_f = np.zeros_like(fit_1_v, dtype=np.bool)

fit_2_v = np.zeros_like(fit_1_v)
fit_2_ve = np.zeros_like(fit_1_v)
fit_2_a = np.zeros_like(fit_1_v)
fit_2_ae = np.zeros_like(fit_1_v)
fit_2_b = np.zeros_like(fit_1_v)
fit_2_f = np.zeros_like(fit_1_v, dtype=np.bool)


def distance_to_angle(z):
    if z.unit == u.km/u.s or z.unit == u.m/u.s:
        return (z.to(u.m/u.s)/solar_circumference_per_degree).value * u.deg/u.s

    if z.unit == u.km/u.s/u.s or z.unit == u.m/u.s/u.s:
        return (z.to(u.m/u.s/u.s)/solar_circumference_per_degree).value * u.deg/u.s/u.s


def angle_to_distance(z):
    if z.unit == u.deg/u.s:
        return (z*solar_circumference_per_degree).value * u.m/u.s

    if z.unit == u.deg/u.s/u.s:
        return (z*solar_circumference_per_degree).value * u.m/u.s/u.s


def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)

root = ''
for value in (nt, dt.value, sigma.value, s0.value, v0.value, a0.value, ntrial):
    root = root + '{:n}'.format(value) + '_'
root += 'fitbias_v_versus_a'
root = clean_for_overleaf(root)


# Acceleration values to try
a = a0 * np.linspace(-1, 1, 3)

# Time range
t = dt * np.arange(0, nt)

# Initial velocity
v = distance_to_angle(v0)

# Go through all the accelerations
for j in range(0, na):
    this_a = distance_to_angle(a[j])

    position = s0 + v*t + 0.5*this_a*t*t

    print(' ')
    print('Acceleration index ', j, na)
    print('True value v ', v)
    print('True value a ', a[j])

    # Go through all the trials
    for i in range(0, ntrial):
        noise = sigma*np.random.normal(loc=0.0, scale=1.0, size=nt)

        fit = FitPosition(t, position + noise, position_error, n_degree=2)
        fit_2_v[j, i] = fit.velocity.value
        fit_2_ve[j, i] = fit.velocity_error.value
        fit_2_a[j, i] = fit.acceleration.value
        fit_2_ae[j, i] = fit.acceleration_error.value
        fit_2_b[j, i] = fit.BIC
        fit_2_f[j, i] = fit.fitted

        z1 = FitPosition(t, position + noise, position_error, n_degree=1)
        fit_1_v[j, i] = z1.velocity.value
        fit_1_ve[j, i] = z1.velocity_error.value
        fit_1_b[j, i] = z1.BIC
        fit_1_f[j, i] = z1.fitted
    print('degree 1 polynomial fit v +/- dv', np.mean(fit_1_v[j, :]), np.std(fit_1_v[j, :]))
    print('degree 2 polynomial fit v +/- dv', np.mean(fit_2_v[j, :]), np.std(fit_2_v[j, :]))
    print('degree 2 polynomial fit a +/- da', np.mean(fit_2_a[j, :]), np.std(fit_2_a[j, :]))

# Convert to distance units
fit_1_v = angle_to_distance(fit_1_v * u.deg/u.s).to(speed_unit).value
# fit_1_ve = (fit_1_ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

fit_2_v = angle_to_distance(fit_2_v * u.deg/u.s).to(speed_unit).value
# fit_2_ve = (fit_2_ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

fit_2_a = angle_to_distance(fit_2_a * u.deg/u.s/u.s).to(acceleration_unit).value
# fit_2_ae = (fit_2_ae * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value

# Save the data
# filename = os.path.expanduser('~/eitwave/dat/test_fitposition/test_fitposition.npz')
# np.savez(filename, z1v, z1ve, fitv, fitve, fita, fitae, sz1b, fit_2_b)

#
# Plotting from hereon down
#

# Each of the accelerations
for i in range(0, na):
    # Acceleration at this index
    a_at_index = a[i].value
    plot_label = plot_labels[i]

    fig, ax = plt.subplots()
    # Do a 2-dimensional histogram of the results, probably the simplest to understand
    # First, do a fit
    aaa = np.ma.array(fit_2_a[i, :], mask=False) #~fit_2_f[i, :])
    aaa_mean = np.nanmean(aaa)
    aaa_std = np.nanstd(aaa)
    print('Mean acceleration = {:n} +/- {:n}'.format(aaa_mean, aaa_std))

    vvv = np.ma.array(fit_2_v[i, :], mask=False) #~fit_2_f[i, :])
    vvv_mean = np.nanmean(vvv)
    vvv_std = np.nanstd(vvv)
    print('Mean velocity = {:n} +/- {:n}'.format(vvv_mean, vvv_std))

    this_poly = np.polyfit(aaa, vvv, 1)
    best_fit = np.polyval(this_poly, a_x)

    hist2d = ax.hist2d(aaa, vvv, bins=[40, 40], range=[[a_x[0], a_x[-1]], [v_y[0], v_y[-1]]])
    ax.set_xlabel('{:s} ({:s})'.format(a_fit, a_string))
    ax.set_ylabel('{:s} ({:s})'.format(v_fit, v_string))
    ax.set_title('{:s} acceleration and velocity fits {:s}'.format(plot_label, subtitle))
    ax.grid(linestyle=":")
    ax.set_xlim(a_x[0], a_x[-1])
    ax.set_ylim(v_y[0], v_y[-1])
    ax.axhline(v0.value, label=v_true + ' ({:n} {:s})'.format(v0.value, v_string), color='red', linestyle="--", zorder=2000)
    ax.axvline(a_at_index, label=a_true + '({:n} {:s})'.format(a_at_index, a_string), color='red', linestyle=":", zorder=2000)
    label_fit = '{:s}={:.0f}{:s} + {:.0f}'.format(v_fit, this_poly[0], a_fit, this_poly[1])
    ax.plot(a_x, best_fit, label='best fit ({:s})'.format(label_fit), color='red')
    cbar = fig.colorbar(hist2d[3])
    cbar.ax.set_ylabel('number')
    plt.legend(framealpha=0.5, loc='lower left', fontsize=11)
    plt.tight_layout()
    if save:
        filename = '{:s}_hist2d_{:n}_{:s}.png'.format(image_root_name, a_at_index, root)
        plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)
    else:
        plt.show()

