import numpy as np
import astropy.units as u


def arc_points(a, b, n=100):

    lat, lon = GreatCircle().waypoints(n=n)

    return SkyCoord()


class GreatCircleArc:
    @u.quantity_input(lon1=u.degree, lat1=u.degree, lon2=u.degree, lat2=u.degree,
                      radius=u.m)
    def __init__(self, lon1, lat1, lon2, lat2, radius):
        """
        Given two points on a sphere (in degrees) and radius of the sphere,
        calculate distance between them and waypoints along the great circle.

        Parameters
        ----------
        lat1 : float
            latitude of point 1 in degrees

        lon1 : float
            longitude of point 1 in degrees

        lat2 : float
            latitude of point 2 in degrees

        lon2 : float
            longitude of point 2 in degrees

        radius : float
            radius of the sphere (e.g. Sun) in meters

        References
        ----------
        The calculation of the great circle and the waypoints is based on
        https://en.wikipedia.org/wiki/Great-circle_navigation .
        """
        self.lon1 = lon1.to(u.degree)
        self.lat1 = lat1.to(u.degree)
        self.lon2 = lon2.to(u.degree)
        self.lat2 = lat2.to(u.degree)
        self.radius = radius

        # Calculate the difference in longitude between the two points.
        lon12 = self.lon2 - self.lon1
        if lon12 > 180*u.degree:
            lon12 -= 360.0*u.degree
        elif lon12 < -180*u.degree:
            lon12 += 360.0*u.degree
        self.lon12_rad = np.deg2rad(lon12.to(u.degree).value)

        # Convert to radians
        self.lat1_rad = np.deg2rad(self.lat1.to(u.degree).value)
        self.lat2_rad = np.deg2rad(self.lat2.to(u.degree).value)
        self.lon1_rad = np.deg2rad(self.lon1.to(u.degree).value)
        self.lon2_rad = np.deg2rad(self.lon2.to(u.degree).value)

        # Calculate a1 and a2
        # TODO what are these quantities?
        tan_a1_numerator = np.sin(self.lon12_rad)
        tan_a1_denominator = ((np.cos(self.lat1_rad)*np.tan(self.lat2_rad)) -
                              (np.sin(self.lat1_rad)*np.cos(self.lon12_rad)))

        tan_a2_numerator = np.sin(self.lon12_rad)
        tan_a2_denominator = ((-np.cos(self.lat2_rad)*np.tan(self.lat1_rad)) +
                              (np.sin(self.lat2_rad)*np.cos(self.lon12_rad)))

        self.a1 = np.arctan2(tan_a1_numerator, tan_a1_denominator)
        self.a2 = np.arctan2(tan_a2_numerator, tan_a2_denominator)

        # Calculate the central angle.  Note that if the central angle is
        # very small (< 0.01 degrees) an alternate formula should be used
        # because the one below is not accurate enough.  This has not been
        # implemented yet.
        cos_central_angle_12 = (np.sin(self.lat1_rad)*np.sin(self.lat2_rad)) + (np.cos(self.lat1_rad)*np.cos(self.lat2_rad)*np.cos(self.lon12_rad))
        self.central_angle_12 = np.arccos(cos_central_angle_12) * u.deg

    def distance(self):
        """
        Calculate the distance on the sphere between the start and end points.

        Returns
        -------
        float
            the distance (measured in the units of the radius) between the
            two points on the sphere.

        """
        return self.central_angle_12.to(u.degree).value * self.radius

    def waypoints(self, n=100):
        """
        Calculate the waypoints between the start and end points.

        Parameters
        ----------
        n : int
            number of waypoints to calculate.

        Returns
        -------
        lat, lon : ndarray, ndarray
            The latitude and longitude of the way points between and including
            the start and endpoints.

        """
        # find parameters at node of great circle (i.e. when it crosses
        # equator).
        a0 = np.arcsin(np.sin(self.a1) * np.cos(self.lat1_rad))

        tan_central_angle_01_numerator = np.tan(self.lat1_rad)
        tan_central_angle_01_denominator = np.cos(self.a1)
        central_angle_01 = np.arctan2(tan_central_angle_01_numerator,
                                      tan_central_angle_01_denominator)

        lon01 = np.arctan2((np.sin(a0) * np.sin(central_angle_01)),
                           np.cos(central_angle_01))

        lon0 = self.lon1_rad - lon01

        # Calculate the waypoints
        d_increments = np.linspace(0, self.distance(), n).to(u.m).value
        central_angle = central_angle_01 + d_increments / self.radius.to(u.m).value

        lat = np.arcsin(np.cos(a0) * np.sin(central_angle))
        lon_minus_lon0 = np.arctan2((np.sin(a0) * np.sin(central_angle)),
                                    np.cos(central_angle))
        lon = lon0 + lon_minus_lon0

        # Put results in the range -180, 180
        lat = np.rad2deg(lat) * u.deg
        above = lat > 180.0*u.deg
        lat[above] -= 360*u.deg
        below = lat < -180*u.deg
        lat[below] += 360*u.deg

        lon = np.rad2deg(lon) * u.deg
        above = lon > 180.0*u.deg
        lon[above] -= 360*u.deg
        below = lon < -180*u.deg
        lon[below] += 360*u.deg

        return lat, lon
