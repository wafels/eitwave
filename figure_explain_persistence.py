#
# Generate an plot that explains the persistence transform
#
import numpy as np
import datacube_tools
import matplotlib.pyplot as plt
import os

plt.ion()
nt = 300
dt = 12
t = 12 * np.arange(0, nt)

# Oscillation
osc_amp = 1.0
osc_per = 300.0

# Linear amplitude
lin_amp = 4.0

# Noise level
noise_amp = 1.5

osc = 1.0 * np.sin(2 * np.pi * t / osc_per)

linear = lin_amp * t / np.max(t)

noise = noise_amp * np.random.normal(size=nt)

data = osc + linear + noise

npdata = np.zeros((1, 1, nt))
npdata[0, 0, :] = data[:]

persistence = datacube_tools.persistence(npdata)


# Make the figure
plt.plot(t, data, label='simulated data $f(t)$')
plt.plot(t, persistence[0,0,:], label='persistence transform $P(t)$')
plt.legend(loc=4, framealpha=0.5)
plt.xlabel('time (seconds)')
plt.ylabel('data (arbitrary units)')
plt.savefig(os.path.expanduser('~/projects/eitwave-paper/persistence_explanation.eps'))


