import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from app.tracks_gen import create_tracks


def plot(n):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    tracks = create_tracks('sphere', 1.0, n)
    xi, yi, zi = np.array(tracks).T[:, 0]
    xk, yk, zk = np.array(tracks).T[:, 1]
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=10, c='r', zorder=10)
    ax.scatter(xk, yk, zk, s=10, c='b', zorder=10)

    plt.show()


def main():
    plot(2000)
    pass


if __name__ == "__main__":
    main()
