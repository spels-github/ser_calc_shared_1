import numpy as np
import pandas as pd
import time
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from app.tracks_gen import create_tracks
from app.models import Model


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # milliseconds
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)


class Device(object):
    """
    Representation of a device as a set of device-specific parameters and experimental data
    """

    def __init__(self, process_node, supply_voltage, path, resistance=1.5e4, capacitance=1e-15):
        """
        Initialize a Device instance
        :param process_node: device process node in nm
        :param supply_voltage: device supply voltage in V
        :param path: path to xlsx file with experimental data
        """
        self.process_node = process_node
        self.supply_voltage = supply_voltage
        self.parameters_set = {}
        df = pd.read_excel(path, sheet_name='Sheet1')
        self.cs_data = df.as_matrix().T[1]
        self.let_data = df.as_matrix().T[0]
        self.rc_parameters = {'resistance': resistance, 'capacitance': capacitance}

    def get_process_node(self):
        return self.process_node

    def get_supply_voltage(self):
        return self.supply_voltage

    def get_experimental_data(self):
        return np.array([self.let_data, self.cs_data])

    def get_parameters(self, model):
        return self.parameters_set[model.get_model_type()]

    def get_rc_parameters(self):
        return self.rc_parameters

    def find_parameters(self, model):
        """
        Perform curve fit to find device-specific parameters of the chosen model
        """
        # TODO add some sort of success check for fitting and return status here
        self.parameters_set[model.get_model_type()] = model.cross_section_fit(self)


class XlsxFile(object):
    """
    xlsx file for logging of simulation results
    """
    def __init__(self, file_path):
        self.path = file_path

    def save_data(self, data, columns=None, sheet_name='Sheet1'):
        pd.DataFrame(data, columns=columns).to_excel(self.path, sheet_name=sheet_name)

    def read_data(self, sheet_name='Sheet1'):
        data_df = pd.read_excel(self.path, sheet_name=sheet_name, index_col=0)
        return data_df.as_matrix().T


class Simulation(object):

    def __init__(self, geometry, accuracy, output_file, physical_parameters):
        """
        Create simulation task with chosen parameters:
        :param geometry: has value "sphere" or "disk" depending on geometry for track generation
                         (tracks originate on surface of the sphere or on the disk)
        :param accuracy: dic with parameters defining accuracy of Monte-Carlo simulation
                         {'trials_count': int, 'particles_count': int, 'let_values_count': int}
        :param output_file: xlsx file for logging of simulation results
        :param physical_parameters: dic with values of physical parameters
                                    {'diff_coefficient': float, 'ambipolar_diff_coefficient': float}
        """
        self.geometry = geometry
        self.accuracy = accuracy
        self.results = []
        self.output_file = output_file
        self.physical_parameters = physical_parameters

    def get_results(self):
        return self.results

    def run_monte_carlo(self, device, model):
        """
        Run Monte Carlo simulation for chosen device and model.
        :param device: instance of class Device
        :param model: instance of class Model
        :return: returns np.array with columns LET value, mean number of upsets, std_dev, cross-section value
        """

        results = []
        for k in range(self.accuracy['trials_count']):
            with Timer() as t:
                tracks = create_tracks(self.geometry, device.get_process_node()*2e-6, self.accuracy['particles_count'])
                # TODO implement angles
                results.append(model.run_trial(device, tracks, self.accuracy['let_values_count']))
                print("finished trial %d" % k)
            print("=> elapsed time: %s s" % t.secs)
        raw_results = np.array(results).T
        # TODO save raw results to file
        mean = np.mean(raw_results, axis=1)
        std_dev = np.std(raw_results, axis=1)
        if self.geometry == 'disk':
            cross_section = (mean * np.pi * ((device.get_process_node() * 2e-6) ** 2)) / \
                            self.accuracy['particles_count']
        else:
            cross_section = (4 * mean * np.pi * ((device.get_process_node() * 2e-6) ** 2)) / \
                            self.accuracy['particles_count']
        self.results = np.array([np.logspace(-3, 2, self.accuracy['let_values_count']), cross_section, mean, std_dev])
        return self.results

    def run_analytical(self, device, model):
        return model.find_iso_cross_section(device, self.accuracy['let_values_count'])


def calculate_ser(sim_results, spectre):
    """
    Calculate SER integral using Simposon's rule by interpolating cross-section values
    at points where spectre is defined
    :param sim_results: np.array containing simulation results
    :param spectre: np.array containing LET values, Integral fluence, and Differential fluence
    :return: SER value
    """
    spectre_temp = spectre
    cut_spectre = np.split(spectre_temp, np.where(spectre_temp[:, 0] > 1e5)[0])[0].T  # remove values for LET>100

    cs_interpolation = interpolate.interp1d(sim_results[0], sim_results[1])
    new_cs = cs_interpolation(cut_spectre[0] / 1e3)

    ser = integrate.simps(np.multiply(cut_spectre[2] * 1e3, new_cs), cut_spectre[0] / 1e3)
    return ser


def main():
    pass

if __name__ == "__main__":
    main()
