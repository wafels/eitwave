import os
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import re
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from statsmodels.robust import mad
from aware5_without_swapping_emission_axis import FitPosition
from aware_constants import solar_circumference_per_degree

# Save to file
save = False

# Show statistic used
show_statistic = False

# Maintain the overall duration of the time series?
ts = {"maintain": True, "accum": 3, "dt": 12*u.s, "nt": 60}

# Where to save the data.
image_directory = os.path.expanduser('~/eitwave/img/test_fitposition')

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

# Initial acceleration
a0 = 0.3*u.km/u.s/u.s
a = (a0/solar_circumference_per_degree).to(u.deg/u.s/u.s)

# Estimated error
error = sigma*np.ones(nt)

# True velocities to use
nv = 50
dv = 10 * u.km/u.s
v0 = 0.0 * u.km/u.s

# Number of trials at each value of the acceleration
ntrial = 100

# Set up some plotting information
pad_inches = 0.05
sigma_string = '$\sigma=${:n}{:s}'.format(sigma.value, sigma.unit.to_string('latex_inline'))
sample_string = '$n_{t}=$'
trial_string = '{:s}{:n}, $\delta t=${:n}{:s}, {:n} trials'.format(sample_string, nt, dt.value, dt.unit.to_string('latex_inline'), ntrial)
subtitle = '\n{:s}, {:s}'.format(sigma_string, trial_string)

if show_statistic:
    statistic_title = [', mean statistic', ', mean statistic',
                       ', median statistic', ', median statistic',
                       ', median statistic']
else:
    statistic_title = ['', '', '', '', '']


def clean_for_overleaf(s, rule='\W+', rep='_'):
    return re.sub(rule, rep, s)

root = ''
for value in (nt, dt.value, sigma.value, s0.value, v0.value, nv, dv.value, a0.value, ntrial):
    root = root + '{:n}'.format(value) + '_'
root += 'vary_velocity'
root = clean_for_overleaf(root)

# Storage for the results
z1v = np.zeros((nv, ntrial))
z1b = np.zeros_like(z1v)
z2v = np.zeros_like(z1v)
z2a = np.zeros_like(z1v)
z2b = np.zeros_like(z1v)

# Velocity values to try
v = ((v0 + dv*np.arange(0, nv))/solar_circumference_per_degree).to(u.deg/u.s)

# Time range
t = dt*np.arange(0, nt)

# Go through all the velocities
for j in range(0, nv):
    position = s0 + v[j]*t + 0.5*a*t*t

    print(' ')
    print('Velocity index ', j, nv)
    print('True value a ', a)
    print('True value v ', v[j])

    # Go through all the trials
    for i in range(0, ntrial):
        noise = sigma*np.random.normal(loc=0.0, scale=1.0, size=nt)
        z1 = FitPosition(t, position + noise, error, n_degree=1)
        z1v[j, i] = z1.velocity.value
        z1b[j, i] = z1.BIC
        z2 = FitPosition(t, position + noise, error, n_degree=2)
        z2v[j, i] = z2.velocity.value
        z2a[j, i] = z2.acceleration.value
        z2b[j, i] = z2.BIC
    print('degree 1 polynomial fit v +/- dv', np.mean(z1v[j, :]), np.std(z1v[j, :]))
    print('degree 2 polynomial fit v +/- dv', np.mean(z2v[j, :]), np.std(z2v[j, :]))
    print('degree 2 polynomial fit a +/- da', np.mean(z2a[j, :]), np.std(z2a[j, :]))

z1v = (z1v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
z2v = (z2v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s).value
z2a = (z2a * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s).value

#
# Mean velocity and acceleration plots
#
v1 = np.mean(z1v, axis=1)
v1e = np.std(z1v, axis=1)

v2 = np.mean(z2v, axis=1)
v2e = np.std(z2v, axis=1)

a2 = np.mean(z2a, axis=1)
a2e = np.std(z2a, axis=1)
vccs = (v * solar_circumference_per_degree).to(u.km/u.s).value

xlim = (np.min(vccs), np.max(vccs))
v_string = v0.unit.to_string('latex_inline')
a_string = a0.unit.to_string('latex_inline')
plt.ion()
plt.close('all')
plt.figure(1)
plt.errorbar(vccs, v1, yerr=v1e, label='polynomial n=1, fit velocity')
plt.errorbar(vccs, v2, yerr=v2e, label='polynomial n=2, fit velocity')
plt.xlim(xlim)
plt.plot(vccs, vccs, label='true velocity ({:n} {:s})'.format(v0.value, v_string), color='r')
plt.xlabel('true velocity ({:s})'.format(v_string))
plt.ylabel('velocity ({:s})'.format(v_string))
plt.title('(a) velocity' + subtitle + statistic_title[0])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'velocity_mean_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

plt.figure(2)
plt.errorbar(vccs, a2, yerr=a2e, label='polynomial n=2, acceleration')
plt.axhline(a0.value, label='true acceleration', color='r')
plt.xlim(xlim)
plt.xlabel('true velocity ({:s})'.format(v_string))
plt.ylabel('fit acceleration ({:s})'.format(a_string))
plt.title('(b) acceleration' + subtitle + statistic_title[1])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_mean_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

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
plt.xlim(xlim)
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
plt.xlim(xlim)
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('fit acceleration ({:s})'.format(a_string))
plt.title('(b) acceleration ' + subtitle + statistic_title[3])
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)

#
# BIC
#
dBIC = z1b - z2b
bic12 = np.median(dBIC, axis=1)
bic12e = mad(dBIC, axis=1, c=1.0)

bic = np.asarray([-6, -2, 0, 2, 6, 10, 20])
bic_color = ['y', 'y', 'g', 'g', 'g', 'g']
bic_alpha = [0.2, 0.1, 0.1, 0.2, 0.3, 0.4]
bic_label = ['n=1 (positive)', 'n=1 (weak)', 'n=2 (weak)', 'n=2 (positive)', 'n=2 (strong)', 'n=2 (very strong)']

plt.figure(5)
plt.axhline(0, label='$\Delta$BIC=0', color='r', linewidth=3)
plt.errorbar(vccs, bic12, yerr=bic12e, label='$BIC_{1} - BIC_{2}$')
plt.grid()
plt.xlabel('true velocity ({:s})'.format(a_string))
plt.ylabel('$\Delta$BIC')
plt.ylim(np.min(bic), np.max(bic))
plt.xlim(xlim)
plt.title('(c) $\Delta$BIC' + subtitle + statistic_title[4])
for i in range(0, len(bic)-1):
    plt.fill_between(vccs, bic[i], bic[i+1], color=bic_color[i], alpha=bic_alpha[i])
for i in range(0, len(bic_label)):
    plt.text(-4.7, 0.5*(bic[i] + bic[i+1]), bic_label[i], bbox=dict(facecolor=bic_color[i], alpha=bic_alpha[i]))
plt.legend(framealpha=0.9, loc='upper right')
plt.tight_layout()
if save:
    filename = 'bic_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)
