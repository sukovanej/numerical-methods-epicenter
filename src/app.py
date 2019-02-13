import os

from numpy import arctan2, arccos, arcsin, cos, sin, array, subtract, deg2rad, rad2deg, multiply, sqrt


HAVERSINE = os.getenv("HAVERSINE") == "1" or False
EARTH_RADIUS = 6371000


class EarthPosition(object):
    def __init__(self, latitude, longitude, time=0, convertToRad=False):
        self.latitude = deg2rad(latitude) if convertToRad else latitude
        self.longitude = deg2rad(longitude) if convertToRad else longitude

        self.time = float(time)

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


def derivative_in_first(fun, x):
    h = 1e-6
    return (fun(EarthPosition(x.latitude + h, x.longitude)) - fun(EarthPosition(x.latitude - h, x.longitude))) / (2 * h)

def derivative_in_second(fun, x):
    h = 1e-6
    return (fun(EarthPosition(x.latitude, x.longitude + h)) - fun(EarthPosition(x.latitude, x.longitude - h))) / (2 * h)


class EpicenterSolver(object):
    def __init__(self, x1, x2, x3):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

    def f1(self, x):
        return self.x1.d(x) / self.x2.d(x) - self.x1.time / self.x2.time

    def f2(self, x):
        return self.x2.d(x) / self.x3.d(x) - self.x2.time / self.x3.time

    def inverse_jacoby(self, x):
        a = derivative_in_first(self.f1, x)
        b = derivative_in_second(self.f1, x)
        c = derivative_in_first(self.f2, x)
        d = derivative_in_second(self.f2, x)

        return multiply(1 / (a * d - b * c), array([[d, -b], [-c, a]]))

    def solve(self, s, n):
        """Solve the system of equation by Newton-Raphson method."""

        def iteration(x):
            inverse_jacobi = self.inverse_jacoby(x)
            f_matrix = array([[self.f1(x)], [self.f2(x)]])
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
            solution = solver.solve(EarthPosition(60.0, -184.0, 44), 1000)
            print(EarthPosition(
                solution.latitude - last_solution.latitude,
                solution.longitude - last_solution.longitude,
                0
            ))

if __name__ == "__main__":
    x1 = EarthPosition(61.601944, -149.117222, 7.5, convertToRad=True) # Palmer, Alaska
    x2 = EarthPosition(39.746944, -105.210833, 23, convertToRad=True) # Golden, Colorado
    x3 = EarthPosition(4.711111, -74.072222, 44, convertToRad=True) # Bogota, Columbia

    solver = EpicenterSolver(x1, x2, x3)
    solution = solver.solve(EarthPosition(70.601944, -149.117222, 7.5), 10)
    print(solution)
    print("distances: "
            + str(solution.d(x1) / 1600) + ", "
            + str(solution.d(x2) / 1600) + ", "
            + str(solution.d(x3) / 1600))

    if os.getenv("STABILITY_TEST") == "1":
        stability_testing(solution)
