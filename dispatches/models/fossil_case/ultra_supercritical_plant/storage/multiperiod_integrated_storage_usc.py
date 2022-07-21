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

"""
This script uses the IDAES multiperiod class to create a steady state
multiperiod model for the integrated ultra-supercritical power plant and
energy storage system. The purpose of this script is to create a multiperiod
model that can be use for market analysis either using a pricetaker assumption
or in a real-time scenario using the double loop framework. The integrated
storage with ultra-supercritical power plant model is used a steady state model
for creating the multiperiod model.
"""

__author__ = "Naresh Susarla and Soraya Rawlings"

from pathlib import Path
try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7


from pyomo.environ import (NonNegativeReals, ConcreteModel, Constraint, Var)
from idaes.apps.grid_integration.multiperiod.multiperiod import (
    MultiPeriodModel)
from dispatches.models.fossil_case.ultra_supercritical_plant.storage import (
    integrated_storage_with_ultrasupercritical_power_plant as usc)
from dispatches.models.fossil_case.ultra_supercritical_plant import storage


def create_usc_model(pmin, pmax):

    # Set bounds for plant power
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 200  # in MW

    max_power = 436  # in MW
    min_power = int(0.65 * max_power)  # 283 in MW

    m = ConcreteModel()

    with resources.path(storage, "initialized_integrated_storage_usc.json") as data_file_path:
        assert data_file_path.is_file()
        m.usc_mp = usc.main(max_power=max_power,
                            load_from_file=str(data_file_path))

    m.usc_mp.fs.plant_min_power_eq = Constraint(
        expr=m.usc_mp.fs.plant_power_out[0] >= min_power
    )
    m.usc_mp.fs.plant_max_power_eq = Constraint(
        expr=m.usc_mp.fs.plant_power_out[0] <= max_power
    )

    m.usc_mp.fs.hxc.heat_duty.setlb(min_storage_heat_duty * 1e6)
    m.usc_mp.fs.hxd.heat_duty.setlb(min_storage_heat_duty * 1e6)

    m.usc_mp.fs.hxc.heat_duty.setub(max_storage_heat_duty * 1e6)
    m.usc_mp.fs.hxd.heat_duty.setub(max_storage_heat_duty * 1e6)

    # Unfix data
    m.usc_mp.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.usc_mp.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.usc_mp.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.usc_mp.fs.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [m.usc_mp.fs.hxd]:
        salt_hxd.inlet_2.unfix()
        salt_hxd.inlet_1.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    for unit in [m.usc_mp.fs.cooler]:
        unit.inlet.unfix()
    m.usc_mp.fs.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    m.usc_mp.fs.hxc.area.fix(1904)
    m.usc_mp.fs.hxd.area.fix(2830)
    m.usc_mp.fs.hxc.outlet_2.temperature[0].fix(831)
    m.usc_mp.fs.hxd.inlet_1.temperature[0].fix(831)
    m.usc_mp.fs.hxd.outlet_1.temperature[0].fix(513.15)

    return m


def create_usc_mp_block(pmin=None, pmax=None):
    print('>>> Creating USC model and initialization for each time period')

    if pmin is None:
        pmin = int(0.65 * 436) + 1
    if pmax is None:
        pmax = 436 + 30

    m = create_usc_model(pmin, pmax)
    b1 = m.usc_mp

    # Add coupling variables
    b1.previous_power = Var(
        domain=NonNegativeReals,
        initialize=300,
        bounds=(pmin, pmax),
        doc="Previous period power (MW)"
        )

    inventory_max = 1e7
    inventory_min = 75000
    tank_max = 6739292           # Units in kg

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
        initialize=tank_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    b1.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=tank_max-inventory_min,
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

    @b1.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return (
            3600*b1.fs.hxc.inlet_2.flow_mass[0] <=
            b1.previous_salt_inventory_cold
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            b1.salt_inventory_hot +
            b1.salt_inventory_cold == b1.fs.salt_amount)

    return m


# The tank level and power output are linked between the contiguous time periods
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.usc_mp.salt_inventory_hot,
             b2.usc_mp.previous_salt_inventory_hot),
            (b1.usc_mp.fs.plant_power_out[0],
             b2.usc_mp.previous_power)]


# The tank level at the end of the last time period must be the same as at the
# beginning of the first time period
def get_usc_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.usc_mp.salt_inventory_hot,
             b2.usc_mp.previous_salt_inventory_hot)]

# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.


def create_multiperiod_usc_model(n_time_points=4, pmin=None, pmax=None):
    """
    Create a multi-period usc_mp cycle object. This object contains a pyomo
    model with a block for each time instance.

    n_time_points: Number of time blocks to create
    """
    multiperiod_usc = MultiPeriodModel(
        n_time_points,
        lambda: create_usc_mp_block(pmin=None, pmax=None),
        get_usc_link_variable_pairs,
        get_usc_periodic_variable_pairs
        )

    # If you have no arguments, you don't actually need to pass in
    # anything. NOTE: building the model will initialize each time block
    multiperiod_usc.build_multi_period_model()
    return multiperiod_usc
