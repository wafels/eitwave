"""
For Figure 5 in the proposal
"""

import os
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))

import re
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import astropy.units as u
from statsmodels.robust import mad
from aware5 import FitPosition
from aware_constants import solar_circumference_per_degree


from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


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

# Initial velocity
v0 = 500*u.km/u.s
v = (v0/solar_circumference_per_degree).to(u.deg/u.s)
v_true = r'$v_{\mbox{true}}$'

# Estimated error
error = sigma*np.ones(nt)

# True accelerations to use
na = 51
da = 0.2 * u.km/u.s/u.s
a0 = -5.0 * u.km/u.s/u.s
a_true = r'$a_{\mbox{true}}$'

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
for value in (nt, dt.value, sigma.value, s0.value, v0.value, na, da.value, a0.value, ntrial):
    root = root + '{:n}'.format(value) + '_'
root = clean_for_overleaf(root)

# Storage for the results
z1v = np.zeros((na, ntrial))
z1b = np.zeros_like(z1v)
z2v = np.zeros_like(z1v)
z2a = np.zeros_like(z1v)
z2b = np.zeros_like(z1v)

# Acceleration values to try
a = ((a0 + da*np.arange(0, na))/solar_circumference_per_degree).to(u.deg/u.s/u.s)

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


"""
fig2, ax2 = z2.plot()
ax2.plot(t.value, position.value, label='true data')
fig2.show()


degrees = [1, 2]
alphas = [1e-4, 1e-3, 1e-2, 1e-1]
for degree in degrees:
    for alpha in alphas:
        est = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha, normalize=True))
        yy = (position+noise).value
        xx = t.value
        est.fit(xx.reshape(nt, 1), yy.reshape(nt, 1))
        coef = est.steps[-1][1].coef_.ravel()
        print(degree, alpha, coef)

plt.ion()
plt.plot(xx, yy)
plt.plot(xx, est.predict(xx[:, np.newaxis]), color='red')
"""
#
# Mean velocity and acceleration plots
#
a_fit = r'$a_{\mbox{fit}}$'
v_fit = r'$v_{\mbox{fit}}$'

v1 = np.mean(z1v, axis=1)
v1e = np.std(z1v, axis=1)

v2 = np.mean(z2v, axis=1)
v2e = np.std(z2v, axis=1)

a2 = np.mean(z2a, axis=1)
a2e = np.std(z2a, axis=1)
accs = (a * solar_circumference_per_degree).to(u.km/u.s/u.s).value

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
    filename = 'velocity_mean_{:s}.png'.format(root)
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
    filename = 'acceleration_mean_{:s}.png'.format(root)
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
    filename = 'bic_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)


#
# Plot the acceleration on one axis and velocity on the other
#
alpha = 0.5
plt.figure(6)
plt.scatter(z2a, z2v, alpha=alpha)
a_index = 30
a_at_index = accs[a_index]
plt.scatter(z2a[a_index, :], z2v[a_index, :], color='k', alpha=1.0, label='fits when {:s}={:n}{:s}'.format(a_true, a_at_index, a_string))
plt.grid()
plt.title('(d) acceleration and velocity fits' + subtitle + statistic_title[4])
plt.xlabel('{:s} ({:s})'.format(a_fit, a_string))
plt.ylabel('{:s} ({:s})'.format(v_fit, v_string))
plt.axhline(v0.value, label='true velocity', color='k', linestyle="--")
plt.axvline(a_at_index, label='example acceleration', color='k', linestyle=":")
plt.legend(framealpha=0.9, loc='lower left', fontsize=11)
plt.tight_layout()


if save:
    filename = 'fit_acceleration_vs_fit_velocity_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight', pad_inches=pad_inches)




