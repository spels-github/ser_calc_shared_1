import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import freeze_support
from app.ser_calculation import Device, Timer
from app.models import Model


def run_fit(model_type, sim_type):
    """
    Test function. Finds parameter values and plots cross-section curves for the given dataset
    :return:
    """
    device = Device(500, 5, '../reference_cs.xlsx')
    model = Model(model_type, sim_type)

    with Timer() as t:
        device.find_parameters(model)
    print("=> elapsed time: %s s" % t.secs)

    par1, par2 = device.get_parameters(model)
    if model_type == 'point' or sim_type == 'analytical':
        print('Found Lc = {0}, LET0 = {1}'.format(par1, par2))
        parameters = [par1*1e4, par2]
    else:
        print('Found R = {0}, L = {1}'.format(par1, par2))
        parameters = [par1 * 1e4, par2 * 1e4]

    LETs = list(np.logspace(0, 2, 51))
    cs_data = model.find_cross_section(LETs, parameters, device.get_supply_voltage())
    x_data, y_data = device.get_experimental_data()
    y_data = np.sqrt(y_data) * 1e4  # transform coordinates for linear curve
    cs_data = np.sqrt(cs_data) * 1e4
    fig, ax1 = plt.subplots()
    plt.gca().set_xscale('log')
    ax1.plot(x_data, y_data, 'bo')
    ax2 = ax1.twinx()
    ax1.plot(LETs, cs_data, 'r-')
    plt.show()


def main():
    # run_fit('voltage', 'analytical')
    run_fit('point', 'monte_carlo')
    pass


if __name__ == "__main__":
    freeze_support()
    main()
