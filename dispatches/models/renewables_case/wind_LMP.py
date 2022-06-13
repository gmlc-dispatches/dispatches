#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
import pyomo.environ as pyo
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.load_parameters import *

design_opt = False


def wind_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=43,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )


def wind_model(wind_resource_config, verbose=False):
    wind_mw = 200

    m = create_model(wind_mw, None, None, None, None, None, wind_resource_config=wind_resource_config, verbose=verbose)
    if design_opt:
        m.fs.windpower.system_capacity.unfix()

    # set_initial_conditions(m, pem_bar * 0.1)
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.windpower.initialize(outlvl=outlvl)
    wind_om_costs(m)

    return m


def wind_optimize(n_time_points, verbose=False):
    # create the multiperiod model object
    wind = MultiPeriodModel(n_time_points=n_time_points,
                            process_model_func=partial(wind_model, verbose=verbose),
                            linking_variable_func=wind_variable_pairs,
                            periodic_variable_func=wind_periodic_variable_pairs)

    wind.build_multi_period_model(wind_resource)

    m = wind.pyomo_model
    blks = wind.get_active_process_blocks()

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*blk.fs.windpower.electricity[0] * 1e-3     # to $/kWh
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost)

    for i in range(n_time_points):
        blk.lmp_signal.set_value(prices_used[i])

    m.wind_cap_cost = pyo.Param(default=1555, mutable=True)

    n_weeks = n_time_points / (7 * 24)

    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks])) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity) +
                          PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)

    opt = pyo.SolverFactory('ipopt')
    wind_gen = []

    opt.solve(m, tee=verbose)
    wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3

    if verbose:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(wind_cap, 2)} MW Wind")

        ax1.set_xlabel('Hour')
        ax1.set_ylabel('kW', )
        ax1.step(hours, wind_gen, label="Wind Generation")
        ax1.tick_params(axis='y', )
        ax1.legend()

        ax2 = ax1.twinx()
        color = 'k'
        ax2.set_ylabel('LMP [$/MWh]', color=color)
        ax2.plot(hours, lmp_array, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.show()

        print("Wind MW: ", wind_cap)
        print("elec Rev $: ", value(sum([blk.profit for blk in blks])))
        print("NPV $:", value(m.NPV))

    return wind_cap, value(sum([blk.profit for blk in blks])), value(m.NPV)


if __name__ == "__main__":
    wind_optimize(n_time_points=7 * 24, verbose=False)

