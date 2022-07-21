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

from . import multiperiod_integrated_storage_usc as mp_usc


pmin = int(0.65 * 436) + 1
pmax = 436 + 30


@pytest.fixture(scope="module")
def usc_model():

    # Build ultra-supercritical plant base model
    m = mp_usc.create_usc_model(pmin, pmax)
    return m


@pytest.mark.unit
def test_usc_model(usc_model):

    assert isinstance(usc_model.usc_mp.fs.plant_min_power_eq, Constraint)
    assert isinstance(usc_model.usc_mp.fs.plant_max_power_eq, Constraint)

    assert usc_model.usc_mp.fs.hxc.area.fixed
    assert value(usc_model.usc_mp.fs.hxc.area) == 1904
    assert usc_model.usc_mp.fs.hxd.area.fixed
    assert value(usc_model.usc_mp.fs.hxd.area) == 2830
    assert usc_model.usc_mp.fs.hxc.outlet_2.temperature[0].fixed
    assert value(usc_model.usc_mp.fs.hxc.outlet_2.temperature[0]) == 831
    assert usc_model.usc_mp.fs.hxd.inlet_1.temperature[0].fixed
    assert value(usc_model.usc_mp.fs.hxd.inlet_1.temperature[0]) == 831
    assert usc_model.usc_mp.fs.hxd.outlet_1.temperature[0].fixed
    assert value(usc_model.usc_mp.fs.hxd.outlet_1.temperature[0]) == 513.15


@pytest.fixture(scope="module")
def mp_model():

    # Build the multi-period usc model
    m = mp_usc.create_usc_mp_block(pmin=pmin, pmax=pmax)
    return m

@pytest.mark.unit
def test_mp_model(mp_model):

    assert isinstance(mp_model.usc_mp.previous_power, Var)
    assert value(mp_model.usc_mp.previous_power) == 300

    assert isinstance(mp_model.usc_mp.previous_salt_inventory_hot, Var)
    assert value(mp_model.usc_mp.previous_salt_inventory_hot) == 75000

    assert isinstance(mp_model.usc_mp.salt_inventory_hot, Var)
    assert value(mp_model.usc_mp.salt_inventory_hot) == 75000

    assert isinstance(mp_model.usc_mp.previous_salt_inventory_cold, Var)
    assert value(mp_model.usc_mp.previous_salt_inventory_cold) == 6739292 - 75000

    assert isinstance(mp_model.usc_mp.salt_inventory_cold, Var)
    assert value(mp_model.usc_mp.salt_inventory_cold) == 6739292 - 75000

    assert isinstance(mp_model.usc_mp.fs.constraint_ramp_down, Constraint)
    assert isinstance(mp_model.usc_mp.fs.constraint_ramp_up, Constraint)
    assert isinstance(mp_model.usc_mp.fs.constraint_salt_inventory_hot, Constraint)
    assert isinstance(mp_model.usc_mp.fs.constraint_salt_maxflow_hot, Constraint)
    assert isinstance(mp_model.usc_mp.fs.constraint_salt_maxflow_cold, Constraint)
    assert isinstance(mp_model.usc_mp.fs.constraint_salt_inventory, Constraint)

@pytest.mark.unit
def test_get_usc_link_variable_pairs():
    assert str('[(blk1.usc_mp.salt_inventory_hot,'
             'blk2.usc_mp.previous_salt_inventory_hot),'
            '(blk1.usc_mp.fs.plant_power_out[0],'
             'blk2.usc_mp.previous_power)]')

@pytest.mark.unit
def test_get_usc_periodic_variable_pairs():
    assert str('[(b1.usc_mp.salt_inventory_hot,'
             'b2.usc_mp.previous_salt_inventory_hot)]')
