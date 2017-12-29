import numpy as np
import scipy.integrate as integrate


def track_charge(track, LET, lc):
    """
    Calculate collected charge for the given track
    :param track:
    :param LET:
    :param lc:
    :return:
    """
    # track = np.array([[0, 0, 0], [0, 0, -2e-4]])
    def point_source_charge(p): return np.exp(-np.linalg.norm(track[0] + p * (track[1] - track[0]))/lc)
    integral, err = integrate.quad(point_source_charge, 0., 1.)
    return 1.03e-10 * LET * np.linalg.norm(track[1]-track[0]) * integral


def main():
    pass


if __name__ == "__main__":
    main()
