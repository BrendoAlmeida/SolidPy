# -*- coding: utf-8 -*-

_author_ = ""
_copyright_ = ""
_license_ = ""

import csv
import numpy as np


class Export:
    @staticmethod
    def evaluate_max_list(parameter_list, time_list):
        """Method for evaluating the max value of a time dependent
        simulation parameter along the time of max

        Args:
            parameter_list (list): time dependent simulation parameter
            time_list (list): time steps for the supplied parameter

        Returns:
            tuple: tuple containing the max value and its time
        """
        max_index = np.argmax(parameter_list)
        max_value = parameter_list[max_index]
        time_of_max = time_list[max_index]
        return max_value, time_of_max

    @staticmethod
    def refine_peak_parabolic(time_list, parameter_list):
        """Three-point parabolic refinement of a sampled peak.

        Fits P = a*t^2 + b*t + c through the three samples bracketing the
        raw argmax and returns the vertex of the parabola when it is a
        genuine maximum (a < 0) and lies within the bracketing interval.
        Otherwise falls back to the raw argmax sample. Useful for
        pressure peaks sampled on a coarse / adaptive mesh, where the
        true maximum sits between samples and argmax underestimates it.

        Returns:
            tuple: (peak_value, peak_time)
        """
        n = len(parameter_list)
        if n < 3:
            return Export.evaluate_max_list(parameter_list, time_list)

        i = int(np.argmax(parameter_list))
        if i == 0 or i == n - 1:
            return Export.evaluate_max_list(parameter_list, time_list)

        t0, t1, t2 = float(time_list[i - 1]), float(time_list[i]), float(time_list[i + 1])
        y0, y1, y2 = (
            float(parameter_list[i - 1]),
            float(parameter_list[i]),
            float(parameter_list[i + 1]),
        )

        # Lagrange denominador comum: ((t1-t0)(t2-t0)(t2-t1)). só zero se
        # houver duas amostras em t idêntico, em que caso não há interpolar.
        denom = (t1 - t0) * (t2 - t0) * (t2 - t1)
        if denom == 0.0:
            return Export.evaluate_max_list(parameter_list, time_list)

        # Direct Lagrange interpolation: fit a*t**2 + b*t + c through
        # (t0,y0),(t1,y1),(t2,y2). Using the standard 3-point form.
        a = ((t2 - t1) * y0 + (t0 - t2) * y1 + (t1 - t0) * y2) / denom
        b = -(
            (t2 ** 2 - t1 ** 2) * y0
            + (t0 ** 2 - t2 ** 2) * y1
            + (t1 ** 2 - t0 ** 2) * y2
        ) / denom
        c = (
            (t2 - t1) * t1 * t2 * y0
            + (t0 - t2) * t2 * t0 * y1
            + (t1 - t0) * t0 * t1 * y2
        ) / denom


        if a >= 0.0:
            # Concave-up or flat: not a maximum — keep the raw argmax.
            return Export.evaluate_max_list(parameter_list, time_list)

        t_vertex = -b / (2.0 * a)
        if not (t0 <= t_vertex <= t2):
            return Export.evaluate_max_list(parameter_list, time_list)

        p_vertex = a * t_vertex ** 2 + b * t_vertex + c
        if not np.isfinite(p_vertex) or p_vertex < y1:
            return Export.evaluate_max_list(parameter_list, time_list)

        return p_vertex, t_vertex

    @staticmethod
    def evaluate_max_variables_list(time, variables):
        max_variable_list = []

        for data_variables in variables:
            max_variable_list.append(Export.evaluate_max_list(data_variables, time))

        return max_variable_list

    @staticmethod
    def raw_simulation_data_export(
        data, filepath: str, header_line, append: bool = False
    ):
        """Adapter method for direct export of simulation solution given
        by solve_ivp

        Args:
            data (list): solution data list to be exported
            filepath (str): filepath to the csv
            header_line (list): list of strings as csv header
            append (bool, optional): boolean option for appending or overwriting
            existing csv. Defaults to False.

        Returns:
            None
        """

        append_boolean = "a" if append else "w"

        with open(filepath, append_boolean, newline="") as file_data:
            solution_writer = csv.writer(file_data)
            solution_writer.writerow(header_line)

            for data_array in zip(
                *data,
            ):
                solution_writer.writerow(data_array)

            return None

    @staticmethod
    def evaluate_mean(data_list):
        """Calculates the mean value of a list ignoring negative spurious data.

        Args:
            data_list (list): list containing values that cannot be negative
            (for instance thrust, absolute pressures, lengths etc)

        Returns:
            float: mean value of positive data.
        """
        sum = 0
        index = 0
        for data in data_list:
            if data >= 0:
                sum += data
                index += 1

        return sum / index
