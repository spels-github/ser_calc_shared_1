import numpy as np
import math
import scipy.integrate as integrate
from scipy.optimize import minimize_scalar
import ctypes
from scipy import LowLevelCallable
from functools import partial, wraps


Da = 25.
D = 12.

lib = ctypes.CDLL('../libintegrand.dll')

lib.i_c.restype = ctypes.c_double
lib.i_c.argtypes = (ctypes.c_double, ctypes.c_void_p)

lib.i_b.restype = ctypes.c_double
lib.i_b.argtype = ctypes.c_void_p


def memoize2(obj):
    cache = obj.cache = {}

    @wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


def round_to_n(x, n):
    fmt = "%%.%de" % n
    return float(fmt % x)


def voltage_amplitude(track, LET, R, L, capacitance=1.0e-15, resistance=1.4e4):
    res = minimize_scalar(lambda time_log: - current_integral(track, time_log, capacitance, resistance, R, L),
                          bounds=(np.log(1e-14), np.log(1e-9)), method='bounded',
                          options={'xatol': 1e-03})
    # print(res)
    track_length = np.linalg.norm(track[1] - track[0])
    # print('Track length = ', track_length)
    return track_length*1.03e-10*LET * current_integral(track, res.x, capacitance, resistance, R, L) / capacitance


def current_integral(track, time_log, capacitance, resistance, R, L):
    # print ("calculating voltage at %e" % math.exp(time_log))
    track_tuple = tuple(map(tuple, track))
    x_data = np.linspace(np.log(0.99e-14), time_log, 50)
    y_data = [current_table(p_i, track_tuple, capacitance, resistance, R, L) for p_i in x_data]
    integral = integrate.simps(y_data, x_data) * math.exp(-math.exp(time_log) / (capacitance * resistance))
    # print ("voltage at %e is %e" % (math.exp(time_log), integral))
    return integral


def current_table(p, track_tuple, capacitance, resistance, R, L):
    p_r = round(p, 2)
    return math.exp(p)*(track_current(math.exp(p_r), track_tuple, R, L) * math.exp(math.exp(p) / (capacitance * resistance)))


def track_current(t, track_tuple, R, L):
    track = np.asarray(track_tuple)
    i_a = 8 * D * (1 + 2*math.sqrt(4*3.1415*t*Da)/L) / ((math.sqrt(4*3.1415*t*Da))**3)
    integral, err = integrate.quad(lambda p: point_source_current(t, p, track, R, L), 0., 1., epsabs=1e-2)
    # print ("        current at %e is\n sci:%E    %e" % (t, integral))
    return i_a * integral


def point_source_current(time, p, track, R, L):
    vector = track[0] + p * (track[1] - track[0])
    z0 = round_to_n(vector[2], 2)
    r0 = round_to_n(math.sqrt(vector[0] ** 2 + vector[1] ** 2), 2)
    ib = i_b(z0, time, L)
    if ib < 1e-30:
        return 0
    return ib * i_c(r0, time, R)


@memoize2
def i_b(z0, t, L):
    pyarr = [L, z0, Da, t]
    arr = (ctypes.c_double * len(pyarr))(*pyarr)
    user_data = ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p)
    # res = math.exp(-((z0**2)/(4*Da*t))-(t*Da*((3.1415/L)**2)))
    # print "i_b is %e at time %e" % (res, t)
    return lib.i_b(user_data)


@memoize2
def i_c(r0, t, R):
    if r0/t > 1000*Da/R:
        return 0
    # f = lambda r: (r/math.sqrt(R**2 - r**2))*math.exp(-((r0**2 + r**2)/(4*Da*t)))*iv(0, r*r0/(2*Da*t))
    pyarr = [R, r0, Da, t]
    arr = (ctypes.c_double * len(pyarr))(*pyarr)
    user_data = ctypes.cast(ctypes.pointer(arr), ctypes.c_void_p)
    func = LowLevelCallable(lib.i_c, user_data)
    integral, err = integrate.quad(func, 0., R, epsrel=1e-5)
    # print "i_c2 is %e at time %e" % (integral, t)
    return integral


def main():
    pass


if __name__ == "__main__":
    main()
