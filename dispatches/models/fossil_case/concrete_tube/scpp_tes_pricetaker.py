# Pyomo imports
from pyomo.environ import (ConcreteModel,
                           Block,
                           Objective,
                           RangeSet,
                           maximize,
                           Constraint)

# IDAES imports
from idaes.core.util import from_json, to_json
from idaes.core.util import get_solver

# DISPATCHES imports
from scpp_concrete_tes import (build_scpp_flowsheet,
                               fix_dof_and_initialize,
                               unfix_dof_for_optimization)


def build_scenario_model(m):
    set_hours = m.set_hours
    set_days = m.set_days
    # set_years = m.parent_block().set_years

    m.period = Block(set_hours, set_days,
                     rule=build_scpp_flowsheet)

    # Connect the initial temperature profiles
    @m.Constraint(set_hours, set_days,
                  m.period[1, 1].fs.tes.period[1].tube_charge.temperature_wall_index)
    def initial_temperature_constraint(blk, t, d, p1, p2):
        if t == 1:
            return Constraint.Skip

        # FIXME: Hard-coding periods (second one) here.
        return (blk.period[t, d].fs.tes.period[1].concrete.init_temperature[p1, p2] ==
                blk.period[t - 1, d].fs.tes.period[2].concrete.temperature[p1, p2])

    return m


def build_stochastic_program(m):
    # Use this function to set up multiple scenarios
    pass


def initialize_model(m):
    blk = ConcreteModel()
    build_scpp_flowsheet(blk)
    fix_dof_and_initialize(blk)

    to_json(blk, fname="initialized_scpp_flowsheet.json")

    for d in m.set_days:
        for t in m.set_hours:
            from_json(m.period[t, d], fname="initialized_scpp_flowsheet.json")

            unfix_dof_for_optimization(m.period[t, d])


def append_objective_function(m):
    LMP = {(1, 1): 74.76927498,
           (2, 1): 69.06309832,
           (3, 1): 68.29710689,
           (4, 1): 68.17900349,
           (5, 1): 68.11404845,
           (6, 1): 68.0849794,
           (7, 1): 68.07800477,
           (8, 1): 68.11518176,
           (9, 1): 65.88403408,
           (10, 1): 55.50713288,
           (11, 1): 42.66843556,
           (12, 1): 26.72945624,
           (13, 1): 21.3099714,
           (14, 1): 24.28040398,
           (15, 1): 26.87753627,
           (16, 1): 26.8105931,
           (17, 1): 27.04827637,
           (18, 1): 25.32874319,
           (19, 1): 30.26194149,
           (20, 1): 49.58981051,
           (21, 1): 60.74922195,
           (22, 1): 70.6046699,
           (23, 1): 81.79030183,
           (24, 1): 83.20018636}

    m.profit = Objective(
        expr=sum(LMP[t, d] * m.period[t, d].fs.net_power_output[0] * 1e-6 -
                 LMP[t, d] * m.period[t, d].fs.discharge_turbine.work_mechanical[0] * 1e-6 -
                 2.11e-9 * 3600 * m.period[t, d].fs.boiler.control_volume.heat[0] -
                 2.11e-9 * 3600 * m.period[t, d].fs.reheater.control_volume.heat[0]
                 for t in m.set_hours for d in m.set_days),
        sense=maximize
    )


if __name__ == '__main__':
    m = ConcreteModel()
    m.set_days = RangeSet(1)
    m.set_hours = RangeSet(24)

    build_scenario_model(m)
    initialize_model(m)

    for d1 in m.set_days:
        m.period[1, d1].fs.tes.period[1].concrete.init_temperature.fix()

    append_objective_function(m)

    get_solver().solve(m, tee=True)

    to_json(m, fname="scpp_pricetaker_solution.json")

    print("End of the run!")
