import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from aware5 import FitPosition
from aware_constants import solar_circumference_per_degree

nt = 20
dt = 36.0*u.s
sigma = 10*u.degree

t = dt*np.arange(0, nt)
s0 = 0*u.degree
v = ((400*u.km/u.s)/solar_circumference_per_degree).to(u.deg/u.s)
error = sigma*np.ones(nt)

na = 20
da = 0.25 * u.km/u.s/u.s
ntrial = 100
z1v = np.zeros((na, ntrial))
z2v = np.zeros_like(z1v)
z2a = np.zeros_like(z1v)
a = (da*np.arange(0, na)/solar_circumference_per_degree).to(u.deg/u.s/u.s)
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
        z2 = FitPosition(t, position + noise, error, n_degree=2)
        z2v[j, i] = z2.velocity.value
        z2a[j, i] = z2.acceleration.value
    print('degree 1 polynomial fit v +/- dv', np.mean(z1v[j, :]), np.std(z1v[j, :]))
    print('degree 2 polynomial fit v +/- dv', np.mean(z2v[j, :]), np.std(z2v[j, :]))
    print('degree 2 polynomial fit a +/- da', np.mean(z2a[j, :]), np.std(z2a[j, :]))

z1v = (z1v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s)
z2v = (z2v * (u.deg/u.s) * solar_circumference_per_degree).to(u.km/u.s)
z2a = (z2a * (u.deg/u.s/u.s) * solar_circumference_per_degree).to(u.km/u.s/u.s)

v1 = np.mean(z1v, axis=1)
v1e = np.std(z1v, axis=1)

v2 = np.mean(z2v, axis=1)
v2e = np.std(z2v, axis=1)

a2 = np.mean(z2a, axis=1)
a2e = np.std(z2a, axis=1)
accs = (a * solar_circumference_per_degree).to(u.km/u.s/u.s)

plt.ion()
plt.close('all')
plt.figure(1)
plt.errorbar(accs.value, v1.value, yerr=v1e.value, label='degree 1, fit velocity')
plt.errorbar(accs.value, v2.value, yerr=v2e.value, label='degree 2, fit velocity')
plt.axhline(v.value, label='true velocity', color='r')
plt.xlabel('acceleration (km/s/s)')
plt.ylabel('velocity (km/s)')
plt.legend(framealpha=0.5, loc='upper left')

plt.figure(2)
plt.errorbar(accs.value, a2.value, yerr=a2e.value, label='degree 2, acceleration')
plt.plot(accs.value, accs.value, label='true acceleration', color='r')
plt.xlabel('acceleration (km/s/s)')
plt.ylabel('acceleration (km/s/s)')
plt.legend(framealpha=0.5, loc='upper left')

