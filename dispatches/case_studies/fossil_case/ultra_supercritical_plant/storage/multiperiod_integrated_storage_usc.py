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

try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7
from idaes.core.util import from_json, to_json
import idaes.logger as idaeslog

from pyomo.environ import (NonNegativeReals, Constraint, Var, ConcreteModel)
from idaes.apps.grid_integration.multiperiod.multiperiod import (
    MultiPeriodModel)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant.storage import (
    integrated_storage_with_ultrasupercritical_power_plant as usc_w_tes)


def create_usc_model(m=None, pmin=None, pmax=None):

    if m is None:
        m = ConcreteModel(name="Integrated Model")

    # Set bounds for plant power
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 200  # in MW

    max_power = 436  # in MW
    min_power = int(0.65 * max_power)  # 283 in MW
    if pmin is None:
        pmin = int(0.65 * 436) + 1
    if pmax is None:
        pmax = 436 + 30

    m = usc.build_plant_model(m)

    # Create a flowsheet, add properties, unit models, and arcs
    m = usc_w_tes.create_integrated_model(m, max_power=max_power)

    # Give all the required inputs to the model
    usc_w_tes.set_model_input(m)

    # Add scaling factor
    usc_w_tes.set_scaling_factors(m)

    # Initialize the model with a sequential initialization and custom
    # Add cost correlations
    m = usc_w_tes.build_costing(m)

    # Add bounds
    usc_w_tes.add_bounds(m)
    

    m.fs.plant_min_power_eq = Constraint(
        expr=m.fs.plant_power_out[0] >= min_power
    )
    m.fs.plant_max_power_eq = Constraint(
        expr=m.fs.plant_power_out[0] <= max_power
    )

    m.fs.hxc.heat_duty.setlb(min_storage_heat_duty * 1e6)
    m.fs.hxd.heat_duty.setlb(min_storage_heat_duty * 1e6)

    m.fs.hxc.heat_duty.setub(max_storage_heat_duty * 1e6)
    m.fs.hxd.heat_duty.setub(max_storage_heat_duty * 1e6)

    # Add coupling variables
    m.fs.previous_power = Var(
        domain=NonNegativeReals,
        initialize=300,
        bounds=(pmin, pmax),
        doc="Previous period power (MW)"
        )

    inventory_max = 1e7
    inventory_min = 75000
    tank_max = 6739292           # Units in kg

    m.fs.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    m.fs.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    m.fs.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=tank_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    m.fs.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=tank_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @m.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            m.fs.previous_power - 60 <=
            m.fs.plant_power_out[0])

    @m.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            m.fs.previous_power + 60 >=
            m.fs.plant_power_out[0])

    @m.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            m.fs.salt_inventory_hot ==
            m.fs.previous_salt_inventory_hot
            + (3600*m.fs.hxc.tube_inlet.flow_mass[0]
               - 3600*m.fs.hxd.shell_inlet.flow_mass[0])
        )

    @m.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return (
            3600*m.fs.hxd.shell_inlet.flow_mass[0] <=
            m.fs.previous_salt_inventory_hot
        )

    @m.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return (
            3600*m.fs.hxc.tube_inlet.flow_mass[0] <=
            m.fs.previous_salt_inventory_cold
        )

    @m.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            m.fs.salt_inventory_hot +
            m.fs.salt_inventory_cold == m.fs.salt_amount)

    return m


def usc_unfix_dof(m):
    # Unfix data
    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.fs.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [m.fs.hxd]:
        salt_hxd.tube_inlet.unfix()
        salt_hxd.shell_inlet.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    for unit in [m.fs.cooler]:
        unit.inlet.unfix()
    m.fs.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    m.fs.hxc.area.fix(1904)
    m.fs.hxd.area.fix(2830)
    m.fs.hxc.tube_outlet.temperature[0].fix(831)
    m.fs.hxd.shell_inlet.temperature[0].fix(831)
    m.fs.hxd.shell_outlet.temperature[0].fix(513.15)
    


def usc_custom_init(m):

    blk = usc.build_plant_model()
    usc.initialize(blk)

    # Create a flowsheet, add properties, unit models, and arcs
    max_power = 436  # in MW
    min_power = int(0.65 * 436)
    pmin = int(0.65 * 436) + 1
    pmax = 436 + 30
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 200  # in MW


    blk = usc_w_tes.create_integrated_model(blk, max_power=max_power)

    # Give all the required inputs to the model
    usc_w_tes.set_model_input(blk)

    # Add scaling factor
    usc_w_tes.set_scaling_factors(blk)

    # Initialize the model with a sequential initialization and custom
    # routines
    usc_w_tes.initialize(blk)

    # Add cost correlations
    blk = usc_w_tes.build_costing(blk)

    # Initialize with bounds
    usc_w_tes.initialize_with_costing(blk)

    blk.fs.plant_min_power_eq = Constraint(
        expr=blk.fs.plant_power_out[0] >= min_power
    )
    blk.fs.plant_max_power_eq = Constraint(
        expr=blk.fs.plant_power_out[0] <= max_power
    )

    blk.fs.hxc.heat_duty.setlb(min_storage_heat_duty * 1e6)
    blk.fs.hxd.heat_duty.setlb(min_storage_heat_duty * 1e6)

    blk.fs.hxc.heat_duty.setub(max_storage_heat_duty * 1e6)
    blk.fs.hxd.heat_duty.setub(max_storage_heat_duty * 1e6)

    if pmin is None:
        pmin = int(0.65 * 436) + 1
    if pmax is None:
        pmax = 436 + 30

    # Add coupling variables
    blk.fs.previous_power = Var(
        domain=NonNegativeReals,
        initialize=300,
        bounds=(pmin, pmax),
        doc="Previous period power (MW)"
        )

    inventory_max = 1e7
    inventory_min = 75000
    tank_max = 6739292           # Units in kg

    blk.fs.previous_salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt at the beginning of the hour (or time period), kg"
        )
    blk.fs.salt_inventory_hot = Var(
        domain=NonNegativeReals,
        initialize=inventory_min,
        bounds=(0, inventory_max),
        doc="Hot salt inventory at the end of the hour (or time period), kg"
        )
    blk.fs.previous_salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=tank_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt at the beginning of the hour (or time period), kg"
        )
    blk.fs.salt_inventory_cold = Var(
        domain=NonNegativeReals,
        initialize=tank_max-inventory_min,
        bounds=(0, inventory_max),
        doc="Cold salt inventory at the end of the hour (or time period), kg"
        )

    @blk.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            blk.fs.previous_power - 60 <=
            blk.fs.plant_power_out[0])

    @blk.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            blk.fs.previous_power + 60 >=
            blk.fs.plant_power_out[0])

    @blk.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            blk.fs.salt_inventory_hot ==
            blk.fs.previous_salt_inventory_hot
            + (3600*blk.fs.hxc.tube_inlet.flow_mass[0]
               - 3600*blk.fs.hxd.shell_inlet.flow_mass[0])
        )

    @blk.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return (
            3600*blk.fs.hxd.shell_inlet.flow_mass[0] <=
            blk.fs.previous_salt_inventory_hot
        )

    @blk.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return (
            3600*blk.fs.hxc.tube_inlet.flow_mass[0] <=
            blk.fs.previous_salt_inventory_cold
        )

    @blk.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            blk.fs.salt_inventory_hot +
            blk.fs.salt_inventory_cold == blk.fs.salt_amount)

    init_model = to_json(blk, return_dict=True)
    from_json(m, sd=init_model)

    return


# The tank level and power output are linked between the contiguous time periods
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [(b1.fs.salt_inventory_hot,
             b2.fs.previous_salt_inventory_hot),
            (b1.fs.plant_power_out[0],
             b2.fs.previous_power)]


# The tank level at the end of the last time period must be the same as at the
# beginning of the first time period
def get_usc_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """
    # return
    return [(b1.fs.salt_inventory_hot,
             b2.fs.previous_salt_inventory_hot)]

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
        n_time_points=n_time_points,
        process_model_func=create_usc_model,
        initialization_func=usc_custom_init,
        unfix_dof_func=usc_unfix_dof,
        linking_variable_func=get_usc_link_variable_pairs,
        flowsheet_options={"pmin": pmin,
                           "pmax": pmax},
        use_stochastic_build=True,
        outlvl=idaeslog.INFO,
        )

    return multiperiod_usc
