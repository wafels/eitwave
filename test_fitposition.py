import os
from matplotlib import rc_file
matplotlib_file = '~/eitwave/eitwave/matplotlibrc_paper1.rc'
rc_file(os.path.expanduser(matplotlib_file))
import re
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from statsmodels.robust import mad
from aware5 import FitPosition
from aware_constants import solar_circumference_per_degree

save = True
image_directory = os.path.expanduser('~/eitwave/img')

nt = 60
dt = 12.0*u.s
sigma = 5*u.degree

t = dt*np.arange(0, nt)
s0 = 0*u.degree
v0 = 500*u.km/u.s
v = (v0/solar_circumference_per_degree).to(u.deg/u.s)
error = sigma*np.ones(nt)

na = 40
da = 0.25 * u.km/u.s/u.s
a0 = -5.0 * u.km/u.s/u.s
ntrial = 100

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

z1v = np.zeros((na, ntrial))
z1b = np.zeros_like(z1v)
z2v = np.zeros_like(z1v)
z2a = np.zeros_like(z1v)
z2b = np.zeros_like(z1v)
a = ((a0 + da*np.arange(0, na))/solar_circumference_per_degree).to(u.deg/u.s/u.s)
for j in range(0, na):
    position = s0 + v*t + 0.5*a[j]*t*t

    print(' ')
    print('Acceleration index ', j, na)
    print('True value v ', v)
    print('True value a ', a[j])

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
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('velocity ({:s})'.format(v_string))
plt.title('(a) velocity' + subtitle + ', mean statistic')
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'velocity_mean_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight')

plt.figure(2)
plt.errorbar(accs, a2, yerr=a2e, label='polynomial n=2, acceleration')
plt.plot(accs, accs, label='true acceleration', color='r')
plt.xlim(np.min(accs), np.max(accs))
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('fit acceleration ({:s})'.format(a_string))
plt.title('(b) acceleration' + subtitle + ', mean statistic')
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_mean_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight')

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
plt.title('(a) velocity' + subtitle + ', median statistic')
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'velocity_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight')

plt.figure(4)
plt.errorbar(accs, a2, yerr=a2e, label='polynomial n=2, acceleration')
plt.plot(accs, accs, label='true acceleration', color='r')
plt.xlim(np.min(accs), np.max(accs))
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('fit acceleration ({:s})'.format(a_string))
plt.title('(b) acceleration ' + subtitle + ', median statistic')
plt.legend(framealpha=0.5, loc='upper left')
plt.grid()
plt.tight_layout()
if save:
    filename = 'acceleration_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight')

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
plt.xlabel('true acceleration ({:s})'.format(a_string))
plt.ylabel('$\Delta$BIC')
plt.ylim(np.min(bic), np.max(bic))
plt.xlim(np.min(accs), np.max(accs))
plt.title('(c) $\Delta$BIC' + subtitle + ', median statistic')
for i in range(0, len(bic)-1):
    plt.fill_between(accs, bic[i], bic[i+1], color=bic_color[i], alpha=bic_alpha[i])
for i in range(0, len(bic_label)):
    plt.text(-4.7, 0.5*(bic[i] + bic[i+1]), bic_label[i], bbox=dict(facecolor=bic_color[i], alpha=bic_alpha[i]))
plt.legend(framealpha=0.9, loc='upper right')
plt.tight_layout()
if save:
    filename = 'bic_median_{:s}.png'.format(root)
    plt.savefig(os.path.join(image_directory, filename), bbox_inches='tight')
