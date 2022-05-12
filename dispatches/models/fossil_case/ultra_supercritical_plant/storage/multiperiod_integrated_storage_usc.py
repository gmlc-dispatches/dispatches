##############################################################################
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
#
##############################################################################

from pyomo.environ import (NonNegativeReals, ConcreteModel, Constraint, Var)
from idaes.apps.grid_integration.multiperiod.multiperiod import (
    MultiPeriodModel)
from dispatches.models.fossil_case.ultra_supercritical_plant.storage import (
    integrated_storage_with_ultrasupercritical_power_plant as usc)


def create_fossil_ies_model(pmin, pmax):

    # Set bounds for plant power
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 200  # in MW

    max_power = 436  # in MW
    min_power = int(0.65 * max_power)  # 283 in MW
    max_power_storage = 30  # in MW
    min_power_storage = 1  # in MW

    # Load from the json file for faster initialization
    load_from_file = 'initialized_integrated_storage_usc.json'

    m = ConcreteModel()
    m.fossil_ies = usc.main(max_power=max_power,
                            load_from_file=load_from_file)

    m.fossil_ies.fs.plant_min_power_eq = Constraint(
        expr=m.fossil_ies.fs.plant_power_out[0] >= min_power
    )
    m.fossil_ies.fs.plant_max_power_eq = Constraint(
        expr=m.fossil_ies.fs.plant_power_out[0] <= max_power
    )
    # Set bounds for discharge turbine
    m.fossil_ies.fs.es_turbine_min_power_eq = Constraint(
        expr=m.fossil_ies.fs.es_turbine.work[0] * (-1e-6) >= min_power_storage
    )
    m.fossil_ies.fs.es_turbine_max_power_eq = Constraint(
        expr=m.fossil_ies.fs.es_turbine.work[0] * (-1e-6) <= max_power_storage
    )

    m.fossil_ies.fs.hxc.heat_duty.setlb(min_storage_heat_duty * 1e6)
    m.fossil_ies.fs.hxd.heat_duty.setlb(min_storage_heat_duty * 1e6)

    m.fossil_ies.fs.hxc.heat_duty.setub(max_storage_heat_duty * 1e6)
    m.fossil_ies.fs.hxd.heat_duty.setub(max_storage_heat_duty * 1e6)

    # Unfix data
    m.fossil_ies.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.fossil_ies.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fossil_ies.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.fossil_ies.fs.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [m.fossil_ies.fs.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    for unit in [m.fossil_ies.fs.cooler]:
        unit.inlet.unfix()
    m.fossil_ies.fs.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    m.fossil_ies.fs.hxc.area.fix(1904)
    m.fossil_ies.fs.hxd.area.fix(2830)
    m.fossil_ies.fs.hxc.outlet_2.temperature[0].fix(831)
    m.fossil_ies.fs.hxd.inlet_1.temperature[0].fix(831)
    m.fossil_ies.fs.hxd.outlet_1.temperature[0].fix(513.15)

    return m


def create_mp_fossil_ies_block(pmin=None, pmax=None):
    print('>>> Creating USC model and initialization for each time period')

    max_power_total = 436 + 29
    min_power_total = int(0.65 * 436) + 1

    m = create_fossil_ies_model(pmin, pmax)
    b1 = m.fossil_ies

    # Add coupling variables
    b1.previous_power = Var(
        domain=NonNegativeReals,
        initialize=300,
        bounds=(min_power_total, max_power_total),
        doc="Previous period power (MW)"
        )

    inventory_max = 1e7
    inventory_min = 75000
    b1.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    b1.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=inventory_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=inventory_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - 60 <=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + 60 >=
            b1.fs.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            b1.salt_inventory_hot ==
            b1.previous_salt_inventory_hot
            + (3600*b1.fs.hxc.inlet_2.flow_mass[0]
               - 3600*b1.fs.hxd.inlet_1.flow_mass[0])
        )

    @b1.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return (
            3600*b1.fs.hxd.inlet_1.flow_mass[0] <=
            b1.previous_salt_inventory_hot
        )

    @m.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return (
            3600*m.fs.hxc.inlet_2.flow_mass[0] <=
            m.fs.previous_salt_inventory_cold[0]
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            b1.salt_inventory_hot +
            b1.salt_inventory_cold == b1.fs.salt_amount)

    return m


# The tank level and power output are linked between time periods
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.fossil_ies.salt_inventory_hot,
             b2.fossil_ies.previous_salt_inventory_hot),
            (b1.fossil_ies.fs.plant_power_out[0],
             b2.fossil_ies.previous_power)]


# The final tank level and power output must be the same as the initial
# tank level and power output state
def get_usc_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.fossil_ies.salt_inventory_hot,
             b2.fossil_ies.previous_salt_inventory_hot)]

# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.


def create_multiperiod_usc_model(n_time_points=4, pmin=None, pmax=None):
    """
    Create a multi-period fossil_ies cycle object. This object contains a pyomo
    model with a block for each time instance.

    n_time_points: Number of time blocks to create
    """
    mp_fossil_ies = MultiPeriodModel(
        n_time_points,
        lambda: create_mp_fossil_ies_block(pmin=None, pmax=None),
        get_usc_link_variable_pairs,
        get_usc_periodic_variable_pairs
        )

    # If you have no arguments, you don't actually need to pass in
    # anything. NOTE: building the model will initialize each time block
    mp_fossil_ies.build_multi_period_model()
    return mp_fossil_ies
