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
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))

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

# Noise level
sigma = 5*u.degree

# Initial displacement
s0 = 0*u.degree

# Initial velocity
v0 = 250*u.km/u.s
v = (v0/solar_circumference_per_degree).to(u.deg/u.s)
v_true = r'$v_{\mbox{true}}$'

# Estimated error
position_error = sigma*np.ones(nt)


# Three accelerations to consider
na = 3
a0 = 5.0 * u.km/u.s/u.s
da = a0
a_true = r'$a_{\mbox{true}}$'

# Number of trials at each value of the acceleration
ntrial = 10000

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


# Acceleration values to try
a = ((a0 + da*np.arange(0, na))/solar_circumference_per_degree).to(u.deg/u.s/u.s)
accs = (a * solar_circumference_per_degree).to(u.km/u.s/u.s).value

# Time range
t = dt*np.arange(0, nt)

# Go through all the accelerations
for j in range(0, na):
    position = s0 + v*t + 0.5*a[j]*t*t

    print(' ')
    print('Acceleration index ', j, na)
    print('True value v ', v)
    print('True value a ', a[j])

    # Go through all the trials
    for i in range(0, ntrial):
        noise = sigma*np.random.normal(loc=0.0, scale=1.0, size=nt)

        fit = FitPosition(t, position + noise, position_error, n_degree=2, fit_method='constrained')
        fit_2_v[j, i] = fit.velocity.value
        fit_2_ve[j, i] = fit.velocity_error.value
        fit_2_a[j, i] = fit.acceleration.value
        fit_2_ae[j, i] = fit.acceleration_error.value
        fit_2_b[j, i] = fit.BIC
        fit_2_f[j, i] = fit.fitted

        z1 = FitPosition(t, position + noise, position_error, n_degree=1, fit_method='constrained')
        fit_1_v[j, i] = z1.velocity.value
        fit_1_ve[j, i] = z1.velocity_error.value
        fit_1_b[j, i] = z1.BIC
        fit_1_f[j, i] = z1.fitted
    print('degree 1 polynomial fit v +/- dv', np.mean(fit_1_v[j, :]), np.std(fit_1_v[j, :]))
    print('degree 2 polynomial fit v +/- dv', np.mean(fit_2_v[j, :]), np.std(fit_2_v[j, :]))
    print('degree 2 polynomial fit a +/- da', np.mean(fit_2_a[j, :]), np.std(fit_2_a[j, :]))

# Convert to distance units
fit_1_v = (fit_1_v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
fit_1_ve = (fit_1_ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

fit_2_v = (fit_2_v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
fit_2_ve = (fit_2_ve * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value

fit_2_a = (fit_2_a * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value
fit_2_ae = (fit_2_ae * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value

# Save the data
# filename = os.path.expanduser('~/eitwave/dat/test_fitposition/test_fitposition.npz')
# np.savez(filename, z1v, z1ve, fitv, fitve, fita, fitae, sz1b, fit_2_b)

#
# Plotting from hereon down
#

# Set up some plotting information
pad_inches = 0.05
sigma_string = '$\sigma=${:n}{:s}'.format(sigma.value, sigma.unit.to_string('latex_inline'))
sample_string = '$n_{t}=$'
trial_string = '{:s}{:n}, $\delta t=${:n}{:s}, {:n} trials'.format(sample_string, nt, dt.value, dt.unit.to_string('latex_inline'), ntrial)
subtitle = '\n{:s}, {:s}'.format(sigma_string, trial_string)
statistic_title = [', mean statistic', ', median statistic',]
a_fit = r'$a_{\mbox{fit}}$'
v_fit = r'$v_{\mbox{fit}}$'

#
# Overleaf puts some conditions on filenames
#
def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)

#
# Define the part of the output filename that contains information on the time
# series used
#
simulation_info = ''
for value in (nt, dt.value, sigma.value, s0.value, v0.value, na, da.value, a0.value, ntrial, ts["accum"], ts["dt"].value, ts["nt"]):
    simulation_info = simulation_info + '{:n}'.format(value) + '_'
simulation_info = clean_for_overleaf(simulation_info)


# Plotting help
v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')

plt.figure(2)
plt.errorbar(accs, a2, yerr=a2e, label='polynomial n=2, acceleration')
plt.plot(accs, accs, label='true acceleration', color='r')
plt.xlim(np.min(accs), np.max(accs))
plt.xlabel('{:s} ({:s})'.format(a_true, a_string))
plt.ylabel('{:s} ({:s})'.format(a_fit, a_string))
plt.title('(b) acceleration' + subtitle + statistic_title[1])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_{:s}_{:s}.png'.format(name, root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)


plot_info = dict()
plot_info[5] = ((a_index_1, '(b)', [0.0, 6.0], [-500, 1500]),
                (a_index_2, '(a)', [-3.0, 3.0], [-500, 1500]))
plot_info[1] = ((a_index_1, '(d)', [0.0, 6.0], [-500, 1500]),
                (a_index_2, '(c)', [-3.0, 3.0], [-500, 1500]))
plot_info[2] = ((a_index_1, '(d)', [0.0, 6.0], [-500, 1500]),
                (a_index_2, '(c)', [-3.0, 3.0], [-500, 1500]))
for a_index, plot_label, xlim, ylim in plot_info[np.int(sigma.value)]:
    a_at_index = accs[a_index]
    xx = fita[a_index, :]
    yy = fitv[a_index, :]
    xerr = fitae[a_index, :]
    yerr = fitve[a_index, :]
    plt.close('all')
    plt.figure(5)
    colors_index = bic_coloring(dBIC[a_index, :], bic_color, bic_alpha)
    plt.errorbar(xx, yy, mfc=[0, 0, 0, 0.0], mec=[0, 0, 0, 0.5],
                 xerr=xerr, yerr=yerr, markersize=2,
                 ecolor=colors_index, fmt='o',
                 label='fits'.format(a_true, a_at_index, a_string))
    plt.grid()
    plt.title('{:s} acceleration and velocity fits {:s}{:s}'.format(plot_label, subtitle, statistic_title[4]))
    plt.xlabel('{:s} ({:s})'.format(a_fit, a_string))
    plt.ylabel('{:s} ({:s})'.format(v_fit, v_string))
    plt.axhline(v0.value, label=v_true + ' ({:n} {:s})'.format(v0.value, v_string), color='b', linestyle="--", zorder=2000)
    plt.axvline(a_at_index, label=a_true + '({:n} {:s})'.format(a_at_index, a_string), color='b', linestyle=":", zorder=2000)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(framealpha=0.5, loc='lower left', fontsize=11)
    plt.tight_layout()
    if save:
        filename = 'single_fit_acceleration_vs_fit_velocity_{:n}_{:s}.png'.format(a_at_index, root)
        plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

    # Create a probability density plot of the results assuming that
    # each result is normally distributed.
    # Define the grid we will calculate results on
    a_offset = 10
    v_offset = 1250
    a_x = np.linspace(-a_offset + a_at_index, a_offset + a_at_index, 100)
    v_y = np.linspace(-v_offset + v0.value, v_offset + v0.value, 101)
    nax = len(a_x)
    nvy = len(v_y)

    ta = fita[a_index, :].flatten()
    tv = fitv[a_index, :].flatten()
    sigma_a = fitae[a_index, :].flatten()
    sigma_v = fitve[a_index, :].flatten()
    summed = np.zeros(shape=(nvy, nax))
    for i in range(0, ntrial):
        a_prob = gaussian(a_x, ta[i], sigma_a[i], prob=True)
        v_prob = gaussian(v_y, tv[i], sigma_v[i], prob=True)
        prob2d = np.tile(a_prob, (nvy, 1))
        prob2d = np.transpose(prob2d) * v_prob
        summed = summed + np.transpose(prob2d)

    summed = summed / np.sum(summed) / ((a_x[1] - a_x[0]) * (v_y[1] - v_y[0]))

for j in (0, 1):
    if j == 0:
        central_tendency = np.median
        central_tendency_kwargs = {"axis": 1}
        error = mad
        error_kwargs = {"axis": 1, "c": 1.0}
        name = 'median'
    else:
        central_tendency = np.mean
        central_tendency_kwargs = {"axis": 1}
        error = np.std
        error_kwargs = {"axis": 1}
        name = 'mean'

    ct_1_v = central_tendency(fit_1_v, **central_tendency_kwargs)
    # v1e = error
    ct_1_ve = error(fit_1_v, **error_kwargs)

    ct_2_v = central_tendency(fit_2_v, **central_tendency_kwargs)
    ct_2_ve = error(fit_2_v, **error_kwargs)

    ct_2_a = central_tendency(fit_2_a, **central_tendency_kwargs)
    ct_2_a = error(fit_2_a, **error_kwargs)



# Each of the accelerations
for i in range(0, na):

    fig, ax = plt.subplots()
    # Do a 2-dimensional histogram of the results, probably the simplest to understand
    # First, do a fit
    xxx = np.ma.array(xx, mask=~fit_2_f[a_index, :])
    yyy = np.ma.array(yy, mask=~fit_2_f[a_index, :])
    this_poly = np.polyfit(xxx, yyy, 1)
    best_fit = np.polyval(this_poly, a_x)
    fig, ax = plt.subplots()
    hist2d = ax.hist2d(xxx, yyy, bins=[40, 40], range=[[a_x[0], a_x[-1]], [v_y[0], v_y[-1]]])
    ax.set_xlabel('{:s} ({:s})'.format(a_fit, a_string))
    ax.set_ylabel('{:s} ({:s})'.format(v_fit, v_string))
    ax.set_title('{:s} acceleration and velocity fits {:s}{:s}'.format(plot_label, subtitle, statistic_title[4]))
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

