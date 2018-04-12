import numpy as np
from scipy.optimize import curve_fit, brentq
from multiprocessing import freeze_support, cpu_count, Process, Queue
from functools import partial
from app.response_voltage import voltage_amplitude
from app.response_point import track_charge


class Model(object):
    """
    All model-dependent methods are implemented in this class
    """
    def __init__(self, model_type, sim_type):
        if model_type == 'point' or model_type == 'voltage':
            self.model_type = model_type
        else:
            raise NotImplementedError('The chosen model type is not implemented')

        if sim_type == 'analytical' or sim_type == 'monte_carlo':
            self.sim_type = sim_type
        else:
            raise NotImplementedError('The chosen simulation type is not implemented')

    def get_model_type(self):
        return self.model_type

    def get_sim_type(self):
        return self.sim_type

    def cross_section_fit(self, device):
        """
        Find model parameters by performing curve fit to exerimenatal data using a chosen model
        :param device: Device class instance
        :return: values of model parameters
        """
        node = device.get_process_node() / 1e3  # convert nm to um
        x_data = device.get_experimental_data()[0]
        y_data = np.sqrt(device.get_experimental_data()[1]) * 1e4  # convert cm2 to um

        def linearized_cross_section(LET, *params):  # transform cross-section(LET) function to linearized form
            cross_section = self.find_cross_section(LET, params, device.get_supply_voltage())
            return np.sqrt(cross_section) * 1e4

        if self.model_type == 'point' or self.sim_type == "analytical":
            parameters, covariance = curve_fit(linearized_cross_section, x_data, y_data, p0=[node, 1],
                                               bounds=([node / 4, 0.01], [node * 4, 60]),
                                               xtol=1e-4, verbose=2, diff_step=0.1)
            parameters[0] = parameters[0] * 1e-4
        elif self.model_type == 'voltage':
            parameters, covariance = curve_fit(linearized_cross_section, x_data, y_data, p0=[node/4, node * 4],
                                               bounds=([node / 16, node / 2], [node * 4, node * 8]),
                                               xtol=1e-4, verbose=2, diff_step=0.01)
            parameters = parameters * 1e-4
        return parameters

    def find_cross_section(self, LET_values, parameters, vdd):
        """
        Calculate cross-section for given  values of LET and model-dependent parameters
        :param LET_values: list or np.array of LET values
        :param parameters: fitting parameters (model-dependent)
        :param model: model type (point or voltage)
        :param vdd: supply voltage
        :return: cross-section value in cm2
        """
        if self.sim_type == "monte_carlo":

            try:
                iterator = iter(LET_values)
            except TypeError:
                LET_values = [LET_values]

            out_q = Queue()
            chunksize = int(np.ceil(len(LET_values) / float(cpu_count() - 1)))
            procs = []

            for i in range(cpu_count() - 1):
                p = Process(
                    target=worker_fitting,
                    args=(LET_values[chunksize * i:chunksize * (i + 1)],
                          out_q, parameters, self.model_type, vdd))
                procs.append(p)
                p.start()

            resultdict = {}
            for i in range(cpu_count() - 1):
                resultdict.update(out_q.get())

            for p in procs:
                p.join()

            radius_values = [resultdict[LET] for LET in LET_values]

        elif self.sim_type == "analytical":

            L = parameters[0] * 1e-4
            LETth = parameters[1]
            LET_values = np.array(LET_values)
            LET_values[LET_values < LETth] = LETth
            if self.model_type == 'point':
                radius_values = (L * (np.log(LET_values / LETth)) ** 0.75) / np.sqrt(np.pi)
            elif self.model_type == 'voltage':
                radius_values = (L * np.log(LET_values / LETth)) / np.pi

        return np.pi * np.array(radius_values) ** 2

    def run_trial(self, device, tracks, let_count):
        """
        Run Monte Carlo trial for the model
        :param device:
        :param tracks:
        :param let_count: number of LET values
        :return:
        """
        if self.model_type == 'point':

            collection_length, threshold_let = device.get_parameters(self)
            charge = partial(track_charge, LET=1, lc=collection_length)
            all_charges = parallel_solve(charge, tracks)
            trial_results = []
            for LET in np.logspace(-3, 2, let_count):
                # calculates the difference between collected charge and critical charge
                all_charges_delta = np.array(all_charges) * LET - collection_length * threshold_let * 1.03e-10
                trial_results.append(len(all_charges_delta[all_charges_delta >= 0]))

        elif self.model_type == 'voltage':

            R, L = device.get_parameters(self)
            capacitance = device.get_rc_parameters()['capacitance']
            resistance = device.get_rc_parameters()['resistance']

            voltage = partial(voltage_amplitude, LET=1, R=R, L=L, capacitance=capacitance,
                              resistance=resistance)
            all_voltages = parallel_solve(voltage, tracks)
            trial_results = []
            for LET in np.logspace(-3, 2, let_count):
                # calculates the difference between voltage amplitude and half of supply voltage
                all_voltages_delta = np.array(all_voltages) * LET - device.supply_voltage / 2
                trial_results.append(len(all_voltages_delta[all_voltages_delta >= 0]))

        return trial_results

    def find_iso_cross_section(self, device, let_count):
        """
        Find cross-section in isotropic field using analytical approximations
        :param device:
        :param let_count: number of LET values
        :return:
        """
        collection_length, threshold_let = device.get_parameters(self)
        LET_values = np.logspace(-3, 2, let_count)
        LET_values[LET_values < threshold_let/2] = threshold_let/2
        if self.model_type == 'point':
            cross_section = (0.26*(collection_length**2) * (np.log(2*LET_values / threshold_let)) ** 2)
        elif self.model_type == 'voltage':
            cross_section = (0.1*(collection_length**2) * (np.log(2*LET_values / threshold_let)) ** 2.42)
        return np.array([np.logspace(-3, 2, let_count), cross_section])


def find_radius(LET, parameters, model, vdd):
    """
    Use Brent optimization method to find the radius of the region where upset condition is met for
    the given parameter values
    :param LET: LET value
    :param parameters: fitting parameters (model-dependent)
    :param model: model type (point or voltage)
    :param vdd: supply voltage
    :return: radius value in centimeters
    """
    if model == 'point':
        Lc = parameters[0] * 1e-4
        LETth = parameters[1]
        f = lambda r: track_charge(np.array([[0, r, 0], [0, r, -3e-4]]), LET, Lc) - Lc * LETth * 1.03e-10
    elif model == 'voltage':
        R = parameters[0] * 1e-4
        L = parameters[1] * 1e-4
        f = lambda r: voltage_amplitude(np.array([[0, r, 0], [0, r, -3e-4]]), LET, R, L) - vdd / 2
    if f(0) <= 0:
        radius = 0
    else:
        radius, convergence = brentq(f, 0, 10e-4, rtol=1e-4, full_output=True)
    return radius


def parallel_solve(response_function, tracks):
    """
    Find response value for each track using multiprocessing
    :param response_function: function describing circuit response for a chosen model
    :param tracks: list of tracks to solve for
    :param chunk_size: size of tracks list chunk, set to prevent memory overflow
    :return: list of response values for each track
    """
    out_q = Queue()
    chunksize = int(np.ceil(len(tracks) / float(cpu_count() - 1)))
    procs = []

    for i in range(cpu_count() - 1):
        p = Process(
            target=worker_solving,
            args=(tracks[chunksize * i:chunksize * (i + 1)],
                  out_q, response_function))
        procs.append(p)
        p.start()

    resultdict = {}
    for i in range(cpu_count() - 1):
        resultdict.update(out_q.get())

    for p in procs:
        p.join()

    responses = [resultdict[str(track)] for track in tracks]
    return responses


def worker_fitting(LET_values, out_q, parameters, model_type, vdd):
    """
    Worker for multiprocessing in cross-section fitting
    :param LET_values:
    :param out_q:
    :param parameters:
    :param model_type:
    :param vdd:
    """
    outdict = {}
    for LET in LET_values:
        outdict[LET] = find_radius(LET, parameters, model_type, vdd)
    out_q.put(outdict)


def worker_solving(tracks, out_q, response_function):
    """
    Worker for multiprocessing in monte carlo trial for isotropic calculation
    :param LET_values:
    :param out_q:
    :param parameters:
    :param model_type:
    :param vdd:
    """
    outdict = {}
    for track in tracks:
        outdict[str(track)] = response_function(track)
    out_q.put(outdict)


def main():
    pass

if __name__ == "__main__":
    freeze_support()
    main()
