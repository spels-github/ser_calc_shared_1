import numpy as np
import math
import random
from numpy.core.umath_tests import inner1d
from scipy.linalg import expm, norm


def sample_spherical(points_count, dimensions=3):
    """
    Return npoints number of points on ndim sphere.
    returns: array [[x1,y1,z1], [x2,y2,z2] ...] of vectors
    """
    vec = np.random.randn(dimensions, points_count)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def sample_sphere(vectors_count, dimensions_count=3):
    """
    Return nvectors number of vectors originating on ndim-dimensional sphere and directed inside.
    returns: arrays [[x1,y1,z1], [x2,y2,z2] ...] of vectors origins and directions
    """
    origin_vectors = sample_spherical(vectors_count, dimensions=dimensions_count)
    direction_vectors = sample_spherical(vectors_count, dimensions=dimensions_count)
    product = inner1d(origin_vectors, direction_vectors)  # row-wise dot product
    indexes = []
    for i in range(vectors_count):
        if product[i] >= 0:
            indexes.append(i)
    origin_vectors = np.delete(origin_vectors, indexes, 0)
    direction_vectors = np.delete(direction_vectors, indexes, 0)
    return origin_vectors, direction_vectors


def rot_matrix(axis, theta):
    """
    Returns rotation matrix for a given axis and angle in 3D
    :param axis:
    :param theta:
    :return:
    """
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))


def sample_2d_disc(points_count):
    """
    Return npoints number of points on a disc.
    returns: np.array [[x1,y1,z1], [x2,y2,z2] ...] of vectors
    """
    points = []
    while len(points) < points_count:
        x = 2 * random.random() - 1
        y = 2 * random.random() - 1
        r2 = x**2 + y**2
        if r2 <= 1.0:
            points.append([x, y, 0.])
    return np.array(points)


def sample_3d_disc(vectors_count, roll=0., tilt=0.):
    """
    Return vectors_count number of vectors originating on ndim-dimensional sphere and directed inside.
    returns: arrays [[x1,y1,z1], [x2,y2,z2] ...] of vectors origins and directions
    """
    tilt_rotation = np.dot(rot_matrix([1, 0, 0], roll), sample_2d_disc(vectors_count).T)
    roll_rotation = np.dot(rot_matrix([0, 1, 0], tilt), tilt_rotation).T
    n = [math.sin(tilt), -math.sin(roll), math.cos(tilt) * math.cos(roll)]
    roll_rotation += n

    return roll_rotation, np.tile(-np.array(n), (vectors_count, 1))


def ray_intersect_sphere(ray, radius, center=np.array([0., 0., 0.])):
    """
    Calculates the intersection point of a ray and a sphere
    :param numpy.array ray: The ray to check.
    :param numpy.array center: The center of the sphere to check against.
    :param float radius: The radius of the sphere to check against.
    :rtype: numpy.array
    :return: Returns two vectors originating at origin at ending at intersection points
             if an intersection occurs. Returns None if no intersection occurs.
    """

    a = ray[1].dot(ray[1])
    b = 2 * ray[1].dot(ray[0] - center)
    c = ray[0].dot(ray[0]) + center.dot(center) - 2 * ray[0].dot(center) - radius**2
    disc = b**2 - 4 * a * c
    if disc < 0:
        return None   # somehow got here once for a ray on the sphere

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    return np.array([ray[0] + t1 * ray[1], ray[0] + t2 * ray[1]])


def select_track(ray, radius):
    """
    Calculates the intersection point of a ray and a bottom half-sphere
    :param numpy.array ray: The ray to check.
    :rtype: numpy.array
    :return: Returns a vector if an intersection occurs.
             Returns None if no intersection occurs.
    """
    points = ray_intersect_sphere(ray, radius)
    if points is None:
        return None
    if points[0, 2] >= 0:
        if points[1, 2] >= 0:
            return None
        else:
            t = - points[0, 2] / (points[1] - points[0])[2]
            return np.array([points[0] + (points[1] - points[0]) * t,
                             points[1]])
    elif points[1, 2] >= 0:
        t = - points[0, 2] / (points[1] - points[0])[2]
        return np.array([points[0],
                         points[0] + (points[1] - points[0]) * t])
    return points


def create_tracks(geo_type, radius, number, roll_angle=0, tilt_angle=0):
    """
    Create a number of tracks inside the bottom half-sphere with a given radius
    originating on a sphere or a disc
    :param geo_type:
    :param radius:
    :param number:
    :param roll_angle:
    :param tilt_angle:
    :return:
    """
    if geo_type == 'disc':
        origins, directions = sample_3d_disc(number, roll=roll_angle, tilt=tilt_angle)
    elif geo_type == 'sphere':
        origins, directions = sample_sphere(number)
    tracks = []
    for i in range(len(origins)):
        ray = np.array([origins[i] * radius, directions[i] * radius])
        track = select_track(ray, radius)
        if track is not None:
            tracks.append(track)
    return tracks


def main():
    pass


if __name__ == "__main__":
    main()

