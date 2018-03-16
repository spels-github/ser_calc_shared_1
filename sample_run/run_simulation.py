import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from app.models import Model
from multiprocessing import freeze_support
from app.ser_calculation import Device, Simulation, XlsxFile, calculate_ser


def run_sim(model_type, sim_type):
    """
    Example function. Finds parameter values and plots a cross-section curve in isotropic field for the given dataset
    :return:
    """
    # Create Device instance using information from DB: process node, supply voltage and experimental results
    # Necessary parameters should be defined by user before simulation
    device = Device(250, 2.5, '../reference_cs.xlsx')

    # Calculation model chosen by user
    model = Model(model_type, sim_type)

    # Additional parameters of calculation model defined by user via GUI, presented values are default ones
    phys = {'diff_coefficient': 12., 'ambipolar_diff_coefficient': 25.}
    accuracy = {'trials_count': 40, 'particles_count': 400000, 'let_values_count': 300}

    # Log file, doesn't have to be xlsx, it's only used now for convenience. Doesn't even have to be a file,
    # can just be pandas DataFrame stored in the DB
    output_file = XlsxFile('../sim_results.xlsx')

    # Step 1: find model parameters for the chosen device
    device.find_parameters(model)

    par1, par2 = device.get_parameters(model)  # parameters should be saved to the DB
    print('Found par1 = {0}, par2 = {1}'.format(par1, par2))

    # Step 2: run simulation to find cross-section dependence on LET in isotropic field
    sim = Simulation('sphere', accuracy, output_file, phys)
    if sim_type == 'monte_carlo':
        results = sim.run_monte_carlo(device, model)
        output_file.save_data(results.T, columns=['LET', 'cross-section', 'mean number', 'std dev'])
    elif sim_type == 'analytical':
        results = sim.run_analytical(device, model)
        output_file.save_data(results.T, columns=['LET', 'cross-section'])

    fig, ax1 = plt.subplots()
    plt.gca().set_xscale('log')
    ax1.plot(results[0], results[1], 'bo')
    plt.show()


def run_ser():
    # load simulation results and spectre from files
    xlsx = XlsxFile('../sim_results.xlsx')
    spectre = np.loadtxt('../LET3.let')
    sim_results = xlsx.read_data()

    # calculate SER value
    ser = calculate_ser(sim_results, spectre)
    print("SER = {0}".format(ser))


def main():
    # run_sim('voltage', 'analytical')
    run_sim('voltage', 'monte_carlo')  # currently, only "point" model is working correctly
    run_ser()
    pass


if __name__ == "__main__":
    freeze_support()
    main()
