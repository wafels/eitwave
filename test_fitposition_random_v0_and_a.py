"""
A number of fit trials are run.  Each trial has a random initial
velocity and acceleration.  The random distribution used is the
uniform distribution within a specified range.  Plots are made illustrating
the
"""

import os
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))

import re
import numpy as np
from numpy.random import uniform, randint
from scipy.stats import spearmanr
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
a_fit = r'$a_{\mbox{fit}}$'
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

# Time range
t = dt*np.arange(0, nt)

vr = [0, 1000]
ar = [-1, 1]

# Go through all the trials
for i in range(0, ntrial):
    #
    nt_random = randint(low=10, high=60)
    t = dt*np.arange(0, nt)
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
plt.ion()
cc = spearmanr(z2a, z2v)
this_poly = np.polyfit(z2a, z2v, 1)

fig, ax = plt.subplots()
# hist2d = ax.hist2d(z2a, z2v, bins=[40, 40])
ax.errorbar(z2a, z2v, xerr=z2ae, yerr=z2ve, linestyle='none')
ax.set_xlabel('{:s} ({:s})'.format(a_fit, a_string))
ax.set_ylabel('{:s} ({:s})'.format(v_fit, v_string))
ax.set_title('acceleration and velocity fits')
ax.grid(linestyle=":")
label_fit = '{:s}={:.0f}{:s} + {:.0f}'.format(v_fit, this_poly[0], a_fit, this_poly[1])
xlim = ax.get_xlim()
a_x = np.linspace(xlim[0], xlim[1], 100)
best_fit = np.polyval(this_poly, a_x)
ax.plot(a_x, best_fit, label='best fit ({:s})'.format(label_fit), color='red')
ax.set_xlim(xlim)
plt.legend(framealpha=0.5, loc='lower left', fontsize=11)
plt.tight_layout()
if save:
    filename = 'single_fit_acceleration_vs_fit_velocity_hist2d_{:n}_{:s}.png'.format(a_at_index, root)
plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

stop


fig, ax = plt.subplots()
hist2d = ax.hist2d(z2a, z2v, bins=[50, 50])

# Need to fit a line through the histogram to illustrate the correlation better


# Where to save the data.
image_directory = os.path.expanduser('~/eitwave/img/test_fitposition_random_v0_and_a')

# Set up some plotting information
pad_inches = 0.05
sigma_string = '$\sigma=${:n}{:s}'.format(sigma.value, sigma.unit.to_string('latex_inline'))
sample_string = '$n_{t}=$'
trial_string = '{:s}{:n}, $\delta t=${:n}{:s}, {:n} trials'.format(sample_string, nt, dt.value, dt.unit.to_string('latex_inline'), ntrial)
subtitle = '\n{:s}, {:s}'.format(sigma_string, trial_string)


def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)

root = ''
for value in (nt, dt.value, sigma.value, s0.value, v0.value, na, da.value, a0.value, ntrial):
    root = root + '{:n}'.format(value) + '_'
root = clean_for_overleaf(root)

if use_median:
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

#
# Mean - or median - velocity and acceleration plots
#
a_fit = r'$a_{\mbox{fit}}$'
v_fit = r'$v_{\mbox{fit}}$'


v1 = central_tendency(z1v, **central_tendency_kwargs)
# v1e = error(z1v, **error_kwargs)
v1e = np.median(z1ve, axis=1)

v2 = central_tendency(z2v, **central_tendency_kwargs)
# v2e = error(z2v, **error_kwargs)
v2e = np.median(z2ve, axis=1)

a2 = central_tendency(z2a, **central_tendency_kwargs)
# a2e = error(z2a, **error_kwargs)
a2e = np.median(z2ae, axis=1)
accs = (a * solar_circumference_per_degree).to(u.km/u.s/u.s).value


# BIC stuff
bic12 = central_tendency(dBIC, **central_tendency_kwargs)
bic12e = error(dBIC, **error_kwargs)
bic = np.asarray([-6, -2, 0, 2, 6, 10, 20])
bic_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
bic_alpha = [0.2, 0.1, 0.1, 0.2, 0.3, 0.4]
bic_label = ['n=1 (positive)', 'n=1 (weak)', 'n=2 (weak)', 'n=2 (positive)', 'n=2 (strong)', 'n=2 (very strong)']

# Plotting help
v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')


plt.ion()
plt.close('all')
plt.figure(1)
plt.errorbar(accs, v1, yerr=v1e, label='polynomial n=1, fit velocity')
plt.errorbar(accs, v2, yerr=v2e, label='polynomial n=2, fit velocity')
plt.xlim(np.min(accs), np.max(accs))
plt.axhline(v0.to(u.km/u.s).value, label='true velocity ({:n} {:s})'.format(v0.value, v_string), color='r')
plt.xlabel('{:s} ({:s})'.format(a_true, a_string))
plt.ylabel('{:s} ({:s})'.format(v_fit, v_string))
plt.title('(a) velocity' + subtitle + statistic_title[0])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'velocity_{:s}_{:s}.png'.format(name, root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)


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

"""
#
# Median velocity and acceleration plots
#
v1 = np.median(z1v, axis=1)
v1e = mad(z1v, axis=1, c=1.0)

v2 = np.median(z2v, axis=1)
v2e = mad(z2v, axis=1, c=1.0)

a2 = np.median(z2a, axis=1)
a2e = mad(z2a, axis=1, c=1.0)

v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')
plt.figure(3)
plt.errorbar(accs, v1, yerr=v1e, label='polynomial n=1, fit velocity')
plt.errorbar(accs, v2, yerr=v2e, label='polynomial n=2, fit velocity')
plt.xlim(np.min(accs), np.max(accs))
plt.axhline(v0.to(u.km/u.s).value, label='true velocity ({:n} {:s})'.format(v0.value, v_string), color='r')
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('velocity ({:s})'.format(v_string))
plt.title('(a) velocity' + subtitle + statistic_title[2])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'velocity_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

plt.figure(4)
plt.errorbar(accs, a2, yerr=a2e, label='polynomial n=2, acceleration')
plt.plot(accs, accs, label='true acceleration', color='r')
plt.xlim(np.min(accs), np.max(accs))
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('fit acceleration ({:s})'.format(a_string))
plt.title('(b) acceleration ' + subtitle + statistic_title[3])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)
"""

plt.figure(3)
plt.axhline(0, label='$\Delta$BIC=0', color='r', linewidth=3)
plt.errorbar(accs, bic12, yerr=bic12e, label='$BIC_{1} - BIC_{2}$')
plt.grid()
plt.xlabel('{:s} ({:s})'.format(a_true, a_string))
plt.ylabel('$\Delta$BIC')
plt.ylim(np.min(bic), np.max(bic))
plt.xlim(np.min(accs), np.max(accs))
plt.title('(c) $\Delta$BIC' + subtitle + statistic_title[4])
for i in range(0, len(bic)-1):
    plt.fill_between(accs, bic[i], bic[i+1], color=bic_color[i], alpha=bic_alpha[i])
for i in range(0, len(bic_label)):
    plt.text(-4.7, 0.5*(bic[i] + bic[i+1]), bic_label[i], bbox=dict(facecolor=bic_color[i], alpha=bic_alpha[i]))
plt.legend(framealpha=0.9, loc='upper right')
plt.tight_layout()
if save:
    filename = 'bic_{:s}_{:s}.png'.format(name, root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)


#
# Plot the acceleration on one axis and velocity on the other for all
# accelerations, highlighting one selection in particular
#
def bic_coloring(dbic, bic_color, bic_alpha):
    dbic_flat = dbic.flatten()
    color = []
    for i in range(0, len(dbic_flat)):
        this = dbic_flat[i]
        if this < -6:
            alpha_at_index = 0.3
            rgb = bic_color[0]
        elif this > 20:
            alpha_at_index = 0.4
            rgb = bic_color[-1]
        else:
            this_index = np.where(bic > this)[0][0] - 1
            alpha_at_index = bic_alpha[this_index]
            rgb = bic_color[this_index]
        color.append([rgb[0], rgb[1], rgb[2], alpha_at_index])
    return color

a_index = 40  # 3 km/s/s
a_index = 25  # 0 km/s/s
a_at_index = accs[a_index]
xx = z2a[a_index, :]
yy = z2v[a_index, :]
xerr = z2ae[a_index, :]
yerr = z2ve[a_index, :]

plt.figure(4)
all_results_colors = bic_coloring(dBIC, bic_color, bic_alpha)
plt.scatter(z2a.flatten(), z2v.flatten(), color=all_results_colors)
plt.scatter(xx, yy, edgecolors='k', facecolors='none', label='fits when {:s}={:n}{:s}'.format(a_true, a_at_index, a_string))
plt.grid()
plt.title('(d) acceleration and velocity fits' + subtitle + statistic_title[4])
plt.xlabel('{:s} ({:s})'.format(a_fit, a_string))
plt.ylabel('{:s} ({:s})'.format(v_fit, v_string))
plt.axhline(v0.value, label=v_true, color='r', linestyle="--", zorder=2000)
plt.axvline(a_at_index, label=a_true, color='r', linestyle=":", zorder=2000)
plt.legend(framealpha=0.5, loc='lower left', fontsize=11)
plt.tight_layout()
if save:
    filename = 'fit_acceleration_vs_fit_velocity_{:n}_{:s}.png'.format(a_at_index, root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)


#
# Plot the acceleration on one axis and velocity on the other for one selection
# in particular.
#
plot_info = dict()
plot_info[5] = ((40, '(b)', [0.0, 6.0], [-500, 1500]),
                (25, '(a)', [-3.0, 3.0], [-500, 1500]))
plot_info[1] = ((40, '(d)', [0.0, 6.0], [-500, 1500]),
                (25, '(c)', [-3.0, 3.0], [-500, 1500]))
plot_info[2] = ((40, '(d)', [0.0, 6.0], [-500, 1500]),
                (25, '(c)', [-3.0, 3.0], [-500, 1500]))
for a_index, plot_label, xlim, ylim in plot_info[np.int(sigma.value)]:
    a_at_index = accs[a_index]
    xx = z2a[a_index, :]
    yy = z2v[a_index, :]
    xerr = z2ae[a_index, :]
    yerr = z2ve[a_index, :]
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
    a_x = np.linspace(-3 + a_at_index, 3 + a_at_index, 100)
    v_y = np.linspace(-500, 1500, 101)
    nax = len(a_x)
    nvy = len(v_y)

    ta = z2a[a_index, :].flatten()
    tv = z2v[a_index, :].flatten()
    sigma_a = z2ae[a_index, :].flatten()
    sigma_v = z2ve[a_index, :].flatten()
    summed = np.zeros(shape=(nax, nvy))
    for i in range(0, ntrial):
        a_prob = gaussian(a_x, ta[i], sigma_a[i], prob=False)
        v_prob = gaussian(v_y, tv[i], sigma_v[i], prob=False)
        prob2d = np.tile(a_prob, (nvy, 1))
        prob2d = np.transpose(prob2d) * v_prob
        summed = summed + prob2d

    # summed = summed/ntrial

    fig, ax = plt.subplots()
    cax = ax.imshow(summed, origin='lower', extent=[a_x.min(), a_x.max(), v_y.min(), v_y.max()], aspect='auto', cmap=cm.bone_r)
    # ax.scatter(ta, tv, 2, color='r', alpha=0.5)
    ax.set_xlabel('{:s} ({:s})'.format(a_fit, a_string))
    ax.set_ylabel('{:s} ({:s})'.format(v_fit, v_string))
    ax.set_title('{:s} acceleration and velocity fits {:s}{:s}'.format(plot_label, subtitle, statistic_title[4]))
    ax.grid(linestyle=":")
    ax.axhline(v0.value, label=v_true + ' ({:n} {:s})'.format(v0.value, v_string), color='k', linestyle="--", zorder=2000)
    ax.axvline(a_at_index, label=a_true + '({:n} {:s})'.format(a_at_index, a_string), color='k', linestyle=":", zorder=2000)
    cbar = fig.colorbar(cax)
    plt.legend(framealpha=0.5, loc='lower left', fontsize=11)
    plt.tight_layout()
    if save:
        filename = 'single_fit_acceleration_vs_fit_velocity__distrib_{:n}_{:s}.png'.format(a_at_index, root)
        plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

    # Scaled difference plots
    diff_a = (ta - a_at_index)/sigma_a
    diff_v = (tv - v0.value)/sigma_v
    fig, ax = plt.subplots()
    ax.hist(diff_a, bins=40, alpha=0.5, histtype="stepfilled", label='acceleration')
    ax.hist(diff_v, bins=40, alpha=0.5, histtype="stepfilled", label='velocity')
    ax.set_xlabel('(fit value - true value)/(estimated error)')
    ax.set_ylabel('number')
    ax.set_title('scaled error distributions')
    ax.grid(linestyle=":")
    plt.legend(framealpha=0.5, fontsize=11)
    plt.tight_layout()
    if save:
        filename = 'single_fit_acceleration_vs_fit_velocity_scaled_distrib_{:n}_{:s}.png'.format(a_at_index, root)
        plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)
