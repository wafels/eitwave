import numpy as np
import sunpy.coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE

# Number of points in great circle
num = 100

# Where is the center?
c = np.asarray([0.0, 0.0, 0.0])

# Load in a map
m = sunpy.map.Map(AIA_171_IMAGE)

# Define the initial and final points
a_coord = SkyCoord(600*u.arcsec, -600*u.arcsec, frame=m.coordinate_frame)
b_coord = SkyCoord(-100*u.arcsec, 800*u.arcsec, frame=m.coordinate_frame)

# Get a Cartesian spatial representation of the points.
# a = a_coord.transform_to('heliocentric').cartesian.xyz.value
# b = b_coord.transform_to('heliocentric').cartesian.xyz.value


def great_circle_arc(a, b, center=SkyCoord(0*u.km, 0*u.km, 0*u.km, frame='heliocentric'), num=100):
    """
    Calculate a user-specified number of points on a great circle arc between a
    start and end point on a sphere.

    :param a:
    :param b:
    :param center:
    :param num:
    :return:
    """
    input_frame = a.frame
    a_unit = a.transform_to('heliocentric').cartesian.xyz.unit
    a_xyz = a.transform_to('heliocentric').cartesian.xyz.value
    b_xyz = b.transform_to('heliocentric').cartesian.xyz.to(a_unit).value
    c_xyz = center.transform_to('heliocentric').cartesian.xyz.to(a_unit).value

    v_xyz = calculate_great_circle_arc(a_xyz, b_xyz, c_xyz, num=num)*a_unit

    return SkyCoord(v_xyz[:, 0], v_xyz[:, 1], v_xyz[:, 2],
                    frame='heliocentric',
                    D0=input_frame.D0,
                    B0=input_frame.B0,
                    L0=input_frame.L0,
                    dateobs=input_frame.dateobs).transform_to(input_frame)


def calculate_great_circle_arc(a, b, c, num=100):
    """
    Calculate a user-specified number of points on a great circle arc between a
    start and end point on a sphere where the start and end points are assumed
    to be x,y,z Cartesian triples on a sphere relative to a center.

    :param a:
    :param b:
    :param c:
    :param num:
    :return:
    """
    x0 = c[0]
    y0 = c[1]
    z0 = c[2]

    # Vector from center to first point
    v1 = np.asarray([a[0] - x0, a[1] - y0, a[2] - z0])

    # Distance of the first point from the center
    r = np.linalg.norm(v1)

    # Vector from center to second point
    v2 = np.asarray([b[0] - x0, b[1] - y0, b[2] - z0])

    # The v3 lies in plane of v1 & v2 and is orthogonal to v1
    v3 = np.cross(np.cross(v1, v2), v1)

    # Ensure that the vector has length r
    v3 = r * v3 / np.linalg.norm(v3)

    # Range through the inner angle between v1 and v2
    inner_angles = np.linspace(0, np.arctan2(np.linalg.norm(np.cross(v1, v2)),
                                             np.dot(v1, v2)), num=num)

    # Calculate the Cartesian locations from the first to second points
    v_xyz = np.zeros(shape=(num, 3))
    for k in range(0, num):
        inner_angle = inner_angles[k]
        v_xyz[k, :] = v1 * np.cos(inner_angle) + v3 * np.sin(inner_angle) + c

    return v_xyz


z = great_circle_arc(a_coord, b_coord)

# Test
print(z[0].transform_to(m.coordinate_frame))
print(z[-1].transform_to(m.coordinate_frame))


fig = plt.figure()
ax = plt.subplot()
m.plot()
ax.plot(z.Tx.value, z.Ty.value)
plt.colorbar()
plt.tight_layout()
plt.show()


"""
#
# Implementing the algorithm of https://www.mathworks.com/matlabcentral/newsreader/view_thread/277881
#
x0 = c[0]
y0 = c[1]
z0 = c[2]

# Vector from center to first point
v1 = np.asarray([a[0]-x0, a[1]-y0, a[2]-z0])

# Distance of the first point from the center
r = np.linalg.norm(v1)

# Vector from center to second point
v2 = np.asarray([b[0]-x0, b[1]-y0, b[2]-z0])

# The v3 lies in plane of v1 & v2 and is orthogonal to v1
v3 = np.cross(np.cross(v1, v2), v1)

# Ensure that the vector has length r
v3 = r*v3/np.linalg.norm(v3)

# Range through the inner angle between v1 and v2
inner_angles = np.linspace(0, np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2)), num=num)

# Calculate the Cartesian locations from the first to second points
vc = np.zeros(shape=(num, 3))
for k in range(0, num):
    inner_angle = inner_angles[k]
    vc[k, :] = v1*np.cos(inner_angle) + v3*np.sin(inner_angle) + c

# Return as co-ordinates
z = SkyCoord(vc[:, 0]*u.km, vc[:, 1]*u.km, vc[:, 2]*u.km, frame='heliocentric',
             D0=m.dsun,
             B0=m.heliographic_latitude, L0=m.heliographic_longitude,
             dateobs=m.date).transform_to(m.coordinate_frame)
"""