import os

from numpy import arctan2, arccos, arcsin, cos, sin, matrix, subtract, deg2rad, rad2deg, multiply, sqrt


HAVERSINE = os.getenv("HAVERSINE") == "1" or False
EARTH_RADIUS = 6371000


class EarthPosition(object):
    def __init__(self, latitude, longitude, time=0):
        self.latitude = deg2rad(latitude)
        self.longitude = deg2rad(longitude)
        self.time = time

    @staticmethod
    def haversine_nb(lon1, lat1, lon2, lat2):
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
        return 6367 * 2 * arcsin(sqrt(a))

    def d(a, b):
        if not HAVERSINE:
            return EARTH_RADIUS * arccos(
                cos(a.latitude) * cos(b.latitude) * cos(a.longitude - b.longitude) + sin(a.latitude) * sin(b.latitude)
            )
        else:
            return a.haversine_nb(a.longitude, a.latitude, b.longitude, b.latitude)

    def __repr__(self):
        return "[{0}, {1}]".format(rad2deg(self.latitude), rad2deg(self.longitude))


def derivative(fun, x):
    h = 1e-6
    return (fun(x + h) - fun(x - h)) / (2 * h)


class EpicenterSolver(object):
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    @classmethod
    def test_function(cls, x1, x2, x):
        return x1.d(x) / x2.d(x) - x1.time / x2.time

    @classmethod
    def longitude_function(cls, x1, x2, latitude):
        return lambda x: cls.test_function(x1, x2, EarthPosition(latitude, x))

    @classmethod
    def latitude_function(cls, x1, x2, longitude):
        return lambda x: cls.test_function(x1, x2, EarthPosition(x, longitude))

    def inverse_jacoby(self, x):
        f1_lat = self.latitude_function(self.x1, self.x2, x.longitude)
        f1_lon = self.longitude_function(self.x1, self.x2, x.latitude)

        f2_lat = self.latitude_function(self.x2, self.x3, x.longitude)
        f2_lon = self.longitude_function(self.x2, self.x3, x.latitude)

        a = derivative(f1_lat, x.latitude)
        b = derivative(f1_lon, x.longitude)
        c = derivative(f2_lat, x.latitude)
        d = derivative(f2_lon, x.longitude)

        return multiply(1 / (a * d - b * c), matrix([[d, -b], [-c, a]]))

    def solve(self, s, n):
        """Solve the system of equation by Newton-Raphson method."""

        def iteration(x):
            inverse_jacobi = self.inverse_jacoby(x)
            f_matrix = matrix([
                [self.test_function(self.x1, self.x2, x)],
                [self.test_function(self.x2, self.x3, x)],
            ])
            product = inverse_jacobi.dot(f_matrix)
            return EarthPosition(x.latitude - product.item(0), x.longitude - product.item(1))

        for _ in range(n):
            s = iteration(s)

        return s


def starting_estimate(x1, x2, x3):
    return EarthPosition(
        (x1.latitude + x2.latitude + x3.latitude) / 3,
        (x1.longitude + x2.longitude + x3.longitude) / 3,
    )


def stability_testing(last_solution):
    for x in range(0, 10):
        for y in range(0, 10):
            x1 = EarthPosition(61.601944 + x * 0.1, -149.117222 + y * 0.1, 7.5) # Palmer, Alaska
            x2 = EarthPosition(39.746944, -105.210833, 23) # Golden, Colorado
            x3 = EarthPosition(4.711111, -74.072222, 44) # Bogota, Columbia
            solver = EpicenterSolver(x1, x2, x3)
            solution = solver.solve(starting_estimate(x1, x2, x3), 1000)
            print(EarthPosition(
                solution.latitude - last_solution.latitude,
                solution.longitude - last_solution.longitude,
                0
            ))

if __name__ == "__main__":
    x1 = EarthPosition(61.601944, -149.117222, 7.5) # Palmer, Alaska
    x2 = EarthPosition(39.746944, -105.210833, 23) # Golden, Colorado
    x3 = EarthPosition(4.711111, -74.072222, 44) # Bogota, Columbia

    solver = EpicenterSolver(x1, x2, x3)
    solution = solver.solve(starting_estimate(x1, x2, x3), 1000)
    print(solution)
    print("distances: "
            + str(solution.d(x1) / 1600) + ", "
            + str(solution.d(x2) / 1600) + ", "
            + str(solution.d(x3) / 1600))

    if os.getenv("STABILITY_TEST") == "1":
        stability_testing(solution)
