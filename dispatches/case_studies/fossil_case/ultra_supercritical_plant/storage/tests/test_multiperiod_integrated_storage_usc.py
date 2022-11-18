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

"""Test for ultra supercritical power plant integrated with charge
storage system

"""

__author__ = "Naresh Susarla"

import pytest
from pyomo.environ import (value, Constraint, Var)

from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant.storage \
     import multiperiod_integrated_storage_usc as mp_usc


pmin = int(0.65 * 436) + 1
pmax = 436 + 30


@pytest.fixture(scope="module")
def multiperiod_model():

    # Build multiperiod integrated power plant base model
    m = None
    model = mp_usc.create_usc_model(m, pmin, pmax)
    return model

@pytest.fixture(scope="module")
def initialized_model():

    # Build ultra-supercritical plant base model
    m = None
    model = mp_usc.create_usc_model(m, pmin, pmax)
    mp_usc.usc_custom_init(model)
    return model

@pytest.mark.unit
def test_usc_model(multiperiod_model):

    assert isinstance(multiperiod_model.fs.plant_min_power_eq, Constraint)
    assert isinstance(multiperiod_model.fs.plant_max_power_eq, Constraint)

    assert isinstance(multiperiod_model.fs.previous_power, Var)
    assert value(multiperiod_model.fs.previous_power) == 300

    assert isinstance(multiperiod_model.fs.previous_salt_inventory_hot, Var)
    assert value(multiperiod_model.fs.previous_salt_inventory_hot) == 75000

    assert isinstance(multiperiod_model.fs.salt_inventory_hot, Var)
    assert value(multiperiod_model.fs.salt_inventory_hot) == 75000

    assert isinstance(multiperiod_model.fs.previous_salt_inventory_cold, Var)
    assert value(multiperiod_model.fs.previous_salt_inventory_cold) == 6739292 - 75000

    assert isinstance(multiperiod_model.fs.salt_inventory_cold, Var)
    assert value(multiperiod_model.fs.salt_inventory_cold) == 6739292 - 75000

    assert isinstance(multiperiod_model.fs.constraint_ramp_down, Constraint)
    assert isinstance(multiperiod_model.fs.constraint_ramp_up, Constraint)
    assert isinstance(multiperiod_model.fs.constraint_salt_inventory_hot, Constraint)
    assert isinstance(multiperiod_model.fs.constraint_salt_maxflow_hot, Constraint)
    assert isinstance(multiperiod_model.fs.constraint_salt_maxflow_cold, Constraint)
    assert isinstance(multiperiod_model.fs.constraint_salt_inventory, Constraint)


@pytest.mark.unit
def test_unfix_dof(multiperiod_model):

    # Verify the degrees of freedom are unfixed and the variables are fixed

    mp_usc.usc_unfix_dof(multiperiod_model)

    assert multiperiod_model.fs.hxc.area.fixed
    assert value(multiperiod_model.fs.hxc.area) == 1904
    assert multiperiod_model.fs.hxd.area.fixed
    assert value(multiperiod_model.fs.hxd.area) == 2830
    assert multiperiod_model.fs.hxc.tube_outlet.temperature[0].fixed
    assert value(multiperiod_model.fs.hxc.tube_outlet.temperature[0]) == 831
    assert multiperiod_model.fs.hxd.shell_inlet.temperature[0].fixed
    assert value(multiperiod_model.fs.hxd.shell_inlet.temperature[0]) == 831
    assert multiperiod_model.fs.hxd.shell_outlet.temperature[0].fixed
    assert value(multiperiod_model.fs.hxd.shell_outlet.temperature[0]) == 513.15

@pytest.mark.unit
def test_custom_initialization(initialized_model):

    # Verify the model structure for the model after custom initialization

    assert isinstance(initialized_model.fs.plant_min_power_eq, Constraint)
    assert isinstance(initialized_model.fs.plant_max_power_eq, Constraint)
    assert isinstance(initialized_model.fs.previous_power, Var)
    assert value(initialized_model.fs.previous_power) == 300

    assert isinstance(initialized_model.fs.previous_salt_inventory_hot, Var)
    assert value(initialized_model.fs.previous_salt_inventory_hot) == 75000

    assert isinstance(initialized_model.fs.salt_inventory_hot, Var)
    assert value(initialized_model.fs.salt_inventory_hot) == 75000

    assert isinstance(initialized_model.fs.previous_salt_inventory_cold, Var)
    assert value(initialized_model.fs.previous_salt_inventory_cold) == 6739292 - 75000

    assert isinstance(initialized_model.fs.salt_inventory_cold, Var)
    assert value(initialized_model.fs.salt_inventory_cold) == 6739292 - 75000

    assert isinstance(initialized_model.fs.constraint_ramp_down, Constraint)
    assert isinstance(initialized_model.fs.constraint_ramp_up, Constraint)
    assert isinstance(initialized_model.fs.constraint_salt_inventory_hot, Constraint)
    assert isinstance(initialized_model.fs.constraint_salt_maxflow_hot, Constraint)
    assert isinstance(initialized_model.fs.constraint_salt_maxflow_cold, Constraint)
    assert isinstance(initialized_model.fs.constraint_salt_inventory, Constraint)

@pytest.mark.unit
def test_get_usc_link_variable_pairs():
    assert str('[(blk1.fs.salt_inventory_hot,'
             'blk2.fs.previous_salt_inventory_hot),'
            '(blk1.fs.plant_power_out[0],'
             'blk2.fs.previous_power)]')
