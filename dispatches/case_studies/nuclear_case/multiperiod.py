#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
__author__ = "Radhakrishna Tumbalam Gooty"


# This file contains utility functions for formulating multi-period models.
# General python imports
import matplotlib.pyplot as plt

# Pyomo imports
from pyomo.environ import (Block,
                           ConcreteModel,
                           check_optimal_termination)
from pyomo.common.timing import TicTocTimer

# IDAES imports
from idaes.core.solvers import get_solver
from idaes.core.util import from_json, to_json


def build_multiperiod_design(m,
                             flowsheet,
                             initialization=None,
                             unfix_dof=None,
                             flowsheet_options={},
                             initialization_options={},
                             unfix_dof_options={},
                             solver=None,
                             verbose=True,
                             stochastic=False,
                             multiyear=False,
                             multiple_days=False,
                             **kwargs):
    """
    This function constructs multiperiod optimization model
    """

    # Create timer object
    timer = TicTocTimer()
    timer.tic("Processing input information.")

    if stochastic:
        # If True, set_scenarios must either be passed as an argument,
        # or it should defined as an attribute of the model
        if "set_scenarios" in kwargs:
            set_scenarios = kwargs["set_scenarios"]
        elif hasattr(m, "set_scenarios"):
            set_scenarios = m.set_scenarios
        else:
            raise Exception(f"stochastic option is set to True, but set_scenarios has "
                            f"not been defined. Either pass set_scenarios as an argument "
                            f"or define it as an attribute of the model.")

    if multiyear:
        # If True, set_years must either be passed as an argument,
        # or it should defined as an attribute of the model
        if "set_years" in kwargs:
            set_years = kwargs["set_years"]
        elif hasattr(m, "set_years"):
            set_years = m.set_years
        else:
            raise Exception(f"multiyear option is set to True, but set_years has "
                            f"not been defined. Either pass set_years as an argument "
                            f"or define it as an attribute of the model.")

    if multiple_days:
        # If True, set_days must either be passed as an argument,
        # or it should defined as an attribute of the model
        if "set_days" in kwargs:
            set_days = kwargs["set_days"]
        elif hasattr(m, "set_days"):
            set_days = m.set_days
        else:
            raise Exception(f"multiple_days option is set to True, but set_days has "
                            f"not been defined. Either pass set_days as an argument "
                            f"or define it as an attribute of the model.")

    # Set of time periods
    if "set_time" in kwargs:
        set_time = kwargs["set_time"]
    elif hasattr(m, "set_time"):
        set_time = m.set_time
    else:
        raise Exception(f"set_time is a required option. Either pass set_time as "
                        f"an argument or define it as an attribute of the model.")

    # Set solver object
    if solver is None:
        solver = get_solver()

    # Construct the set of time periods
    if multiyear and multiple_days:
        set_period = [(t, d, y) for y in set_years for d in set_days for t in set_time]
    elif multiple_days:
        set_period = [(t, d) for d in set_days for t in set_time]
    else:
        set_period = [t for t in set_time]

    """
    Period rule
    """
    timer.toc("Beginning the formulation of the multiperiod problem.")

    def _period_model_rule(options, verbose_flag):

        def _period_model(blk):
            if verbose_flag:
                print("Constructing flowsheet model for ", blk.name)

            flowsheet(blk, options=options)

        return _period_model

    def _build_scenario_model(blk):
        blk.period = Block(set_period, rule=_period_model_rule(flowsheet_options, verbose))

        return blk

    # Construct the multiperiod model
    if stochastic:
        m.scenario = Block(set_scenarios, rule=_build_scenario_model)
    else:
        _build_scenario_model(m)

    timer.toc("Completed the formulation of the multiperiod problem")
    """
    Initialization routine
    """
    if initialization is None:
        print("*** WARNING *** Initialization function is not provided. "
              "Returning the multiperiod model without initialization.")
        return

    b = ConcreteModel()
    flowsheet(b, options=flowsheet_options)
    initialization(b, options=initialization_options)

    result = solver.solve(b)

    try:
        assert check_optimal_termination(result)
    except AssertionError:
        print(f"Flowsheet did not converge to optimality "
              f"after fixing the degrees of freedom.")
        raise

    # Save the solution in json file
    to_json(b, fname="temp_initialized_model.json")
    timer.toc("Created an instance of the flowsheet and initialized it.")

    # Initialize the multiperiod optimization model
    if stochastic:
        for s in set_scenarios:
            for p in set_period:
                from_json(m.scenario[s].period[p], fname="temp_initialized_model.json")

    else:
        for p in set_period:
            from_json(m.period[p], fname="temp_initialized_model.json")

    timer.toc("Initialized the entire multiperiod optimization model.")

    """
    Unfix the degrees of freedom in each period model for optimization model
    """
    if unfix_dof is None:
        print("*** WARNING *** unfix_dof function is not provided. "
              "Returning the model without unfixing degrees of freedom")
        return

    if stochastic:
        for s in set_scenarios:
            for p in set_period:
                unfix_dof(m.scenario[s].period[p], options=unfix_dof_options)

    else:
        for p in set_period:
            unfix_dof(m.period[p], options=unfix_dof_options)

    timer.toc("Unfixed the degrees of freedom from each period model.")


def plot_lmp_signal(time=None,
                    lmp=None,
                    duplicate=True,
                    x_range=None,
                    y_range=None):
    if lmp is None:
        raise Exception("LMP data is not provided!")

    if len(lmp) > 6:
        raise Exception("Number of LMP signals provided exceeds six: the maximum "
                        "number of subplots the function can handle.")

    plt_time = {}
    plt_lmp = {}

    if duplicate:
        for i in lmp:
            plt_time[i] = []
            plt_lmp[i] = []

            for index, value in enumerate(lmp[i]):
                plt_lmp[i].extend([value, value])

                if time is None:
                    plt_time[i].extend([index, index + 1])
                else:
                    plt_time[i].extend([time[i][index] - 1, time[i][index]])

    else:
        for i in lmp:
            plt_lmp[i] = lmp[i]

            if time is None:
                plt_time[i] = [j for j in range(1, 25)]
            else:
                plt_time[i] = time[i]

    grid_shape = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    grid = grid_shape[len(lmp)]

    color = {1: 'tab:red', 2: 'tab:red', 3: 'tab:red',
             4: 'tab:red', 5: 'tab:red', 6: 'tab:red'}

    fig = plt.figure()

    for i in range(1, len(lmp) + 1):
        ax = fig.add_subplot(grid[0], grid[1], i)
        ax.set_xlabel('time (hr)')
        ax.set_ylabel('LMP ($/MWh)')
        ax.plot(plt_time[i], plt_lmp[i], color=color[i])

        if x_range is not None and i in x_range:
            ax.set_xlim(x_range[i][0], x_range[i][1])

        if y_range is not None and i in y_range:
            ax.set_ylim(y_range[i][0], y_range[i][1])

    fig.tight_layout()
    plt.show()


def plot_lmp_and_schedule(time=None,
                          lmp=None,
                          schedule=None,
                          y_label=None,
                          x_range=None,
                          lmp_range=None,
                          y_range=None,
                          x_label="time (hr)",
                          lmp_label="LMP ($/MWh)",
                          color=None,
                          duplicate=True):
    if lmp is None:
        raise Exception("LMP data is not provided!")

    if schedule is None:
        raise Exception("Optimal schedule data is not provided!")

    if len(schedule) > 4:
        raise Exception("len(schedule) exceeds four: the maximum "
                        "number of subplots the function can handle.")

    key_list = {index + 1: value for index, value in enumerate(schedule)}

    plt_time = []
    plt_lmp = []

    for index, value in enumerate(lmp):
        plt_lmp.extend([value, value])

        if time is None:
            plt_time.extend([index, index + 1])
        else:
            plt_time.extend([time[index] - 1, time[index]])

    plt_schedule = {}

    if duplicate:
        for i in schedule:
            plt_schedule[i] = []

            for value in schedule[i]:
                plt_schedule[i].extend([value, value])

    else:
        for i in schedule:
            plt_schedule[i] = schedule[i]

    grid_shape = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    grid = grid_shape[len(schedule)]

    lmp_color = 'tab:red'
    if color is None:
        plt_color = {1: 'tab:blue', 2: 'magenta', 3: 'tab:green', 4: 'tab:cyan'}
    else:
        plt_color = {index + 1: color[value] for index, value in enumerate(color)}

    fig = plt.figure()

    for i in range(1, len(schedule) + 1):
        ax = fig.add_subplot(grid[0], grid[1], i)
        ax.set_xlabel(x_label)
        ax.set_ylabel(lmp_label, color=lmp_color)
        ax.plot(plt_time, plt_lmp, color=lmp_color)
        ax.tick_params(axis='y', labelcolor=lmp_color)

        if x_range is not None:
            ax.set_xlim(x_range[0], x_range[1])

        if lmp_range is not None:
            ax.set_ylim(lmp_range[0], lmp_range[1])

        ax1 = ax.twinx()
        ax1.plot(plt_time, plt_schedule[key_list[i]], color=plt_color[i])
        ax1.tick_params(axis='y', labelcolor=plt_color[i])

        if y_label is not None and key_list[i] in y_label:
            ax1.set_ylabel(y_label[key_list[i]], color=plt_color[i])

        if y_range is not None and key_list[i] in y_range:
            ax1.set_ylim(y_range[key_list[i]][0], y_range[key_list[i]][1])

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    import json

    with open("lmp_price_signal.json") as fp:
        lmp_data = json.load(fp)

    lmp = {1: [lmp_data["0"]["2022"]["1"][str(i)] for i in range(1, 25)],
           2: [lmp_data["0"]["2022"]["8"][str(i)] for i in range(1, 25)]}

    plot_lmp_signal(lmp=lmp, x_range={1: (0, 24), 2: (0, 24)})
