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

"""This is a Generalized Disjunctive Programming model for the
conceptual design of an ultra-supercritical coal-fired power plant
integrated with a charge storage system

"""

__author__ = "Soraya Rawlings and Naresh Susarla"

import logging

# Import Python libraries
from math import pi
from IPython import embed

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective,
                           TransformationFactory, SolverFactory,
                           Expression, value, log, exp, Var)
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.gdp import Disjunct, Disjunction
from pyomo.network.plugins import expand_arcs
from pyomo.contrib.fbbt.fbbt import  _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression

# Import IDAES libraries
import idaes.core.util.unit_costing as icost
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import (HeatExchanger,
                                      MomentumMixingType,
                                      Heater,
                                      Mixer,
                                      PressureChanger)
from idaes.models.unit_models.separator import (Separator,
                                                SplittingType)
from idaes.models.unit_models.heat_exchanger import (delta_temperature_underwood_callback,
                                                     HeatExchangerFlowPattern)
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmIsentropicCompressor,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)

# Import ultra-supercritical power plant model
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

# Import properties package for storage materials
from dispatches.models.fossil_case.properties import (solarsalt_properties,
                                                      hitecsalt_properties,
                                                      thermaloil_properties)

scaling_obj = 1e-7

def create_charge_model(m, add_efficiency=None, power_max=None):
    """Create flowsheet and add unit models

    """

    # Create a block to add charge storage model
    m.fs.charge = Block()

    # Add model data
    _add_data(m)

    # Add molten salt and thermal oil properties
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()
    m.fs.hitec_salt_properties = hitecsalt_properties.HitecsaltParameterBlock()
    m.fs.therminol66_properties = thermaloil_properties.ThermalOilParameterBlock()

    ###########################################################################
    #  Add unit models
    ###########################################################################

    # Add connector, defined as a dummy heat exchanger with Q=0 and a
    # deltaP=0
    m.fs.charge.connector = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": False
        }
    )

    # Add cooler to ensure the outlet of charge heat exchanger is a
    # subcooled liquid before mixing it with the plant
    m.fs.charge.cooler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    # Add pump to increase the pressure of the water to allow mixing
    # it at a desired location within the plant
    m.fs.charge.hx_pump = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )

    # Add recycle mixer
    m.fs.charge.recycle_mixer = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["from_bfw_out", "from_hx_pump"],
            "property_package": m.fs.prop_water,
        }
    )

    ###########################################################################
    #  Declare disjuncts
    ###########################################################################
    # Disjunction 1 for the storage material selection consists of 2 disjuncts:
    #   1. solar_salt_disjunct  ======> Solar salt used as the storage material
    #   2. hitec_salt_disjunct  ======> Hitec salt used as the storage material
    #   3. thermal_oil_disjunct ======> thermal oil used as the storage material
    # Disjunction 2 for the steam source selection consists of 2 disjuncts:
    #   1. vhp_source_disjunct ===> high pressure steam for heat source
    #   2. hp_source_disjunct  ===> intermediate pressure steam for heat source

    # Declare disjuncts in disjunction 1
    m.fs.charge.solar_salt_disjunct = Disjunct(
        rule=solar_salt_disjunct_equations)
    m.fs.charge.hitec_salt_disjunct = Disjunct(
        rule=hitec_salt_disjunct_equations)
    m.fs.charge.thermal_oil_disjunct = Disjunct(
        rule=thermal_oil_disjunct_equations)

    # Declare disjuncts in disjunction 2
    m.fs.charge.vhp_source_disjunct = Disjunct(
        rule=vhp_source_disjunct_equations)
    m.fs.charge.hp_source_disjunct = Disjunct(
        rule=hp_source_disjunct_equations)

    ###########################################################################
    #  Add constraints
    ###########################################################################

    _make_constraints(m, add_efficiency=add_efficiency, power_max=power_max)

    ###########################################################################
    #  Create stream arcs
    ###########################################################################

    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)

    return m

def _add_data(m):
    """Add data to the model
    """

    # Add Chemical engineering cost index for 2019
    m.CE_index = 607.5

    # Add operating hours
    m.number_hours_per_day = 6
    m.fs.charge.hours_per_day = pyo.Param(
        initialize=m.number_hours_per_day,
        doc='Number of hours of charging per day'
    )

    # Define number of years over which the costs are annualized
    m.number_of_years = 30
    m.fs.charge.num_of_years = pyo.Param(
        initialize=m.number_of_years,
        doc='Number of years for cost annualization')

    # Add data to compute overall heat transfer coefficient for the
    # Solar salt, Hitec salt, and Thermal oil storage heat exchangers
    # using the Sieder-Tate Correlation. Parameters for tube diameter
    # and thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985
    m.fs.charge.data_hxc = {
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }
    m.fs.charge.hxc_tube_inner_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['tube_inner_dia'],
        doc='Tube inner diameter [m]')
    m.fs.charge.hxc_tube_outer_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['tube_outer_dia'],
        doc='Tube outer diameter [m]')
    m.fs.charge.hxc_k_steel = pyo.Param(
        initialize=m.fs.charge.data_hxc['k_steel'],
        doc='Thermal conductivity of steel [W/mK]')
    m.fs.charge.hxc_n_tubes = pyo.Param(
        initialize=m.fs.charge.data_hxc['number_tubes'],
        doc='Number of tubes')
    m.fs.charge.hxc_shell_inner_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['shell_inner_dia'],
        doc='Shell inner diameter [m]')

    # Calculate sectional area of storage heat exchanger
    m.fs.charge.hxc_tube_cs_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.charge.hxc_tube_inner_dia**2),
        doc="Tube inside cross sectional area in m2")
    m.fs.charge.hxc_tube_out_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.charge.hxc_tube_outer_dia**2),
        doc="Tube cross sectional area including thickness in m2")
    m.fs.charge.hxc_shell_eff_area = pyo.Expression(
        expr=(
            (pi / 4) *
            m.fs.charge.hxc_shell_inner_dia**2 -
            m.fs.charge.hxc_n_tubes *
            m.fs.charge.hxc_tube_out_area
        ),
        doc="Effective shell cross sectional area in m2")

    m.fs.charge.hxc_tube_dia_ratio = (
        m.fs.charge.hxc_tube_outer_dia / m.fs.charge.hxc_tube_inner_dia)
    m.fs.charge.hxc_log_tube_dia_ratio = log(m.fs.charge.hxc_tube_dia_ratio)

    # Add fuel and storage material cost data
    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'solar_salt_price': 0.49,
        'hitec_salt_price': 0.93,
        'thermal_oil_price': 6.72,
        'storage_tank_material': 3.5,
        'storage_tank_insulation': 235,
        'storage_tank_foundation': 1210
    }
    m.fs.charge.coal_price = pyo.Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.charge.cooling_price = pyo.Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')
    m.fs.charge.solar_salt_price = pyo.Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')
    m.fs.charge.hitec_salt_price = pyo.Param(
        initialize=m.data_cost['hitec_salt_price'],
        doc='Hitec salt price in $/kg')
    m.fs.charge.thermal_oil_price = pyo.Param(
        initialize=m.data_cost['thermal_oil_price'],
        doc='Thermal oil price in $/kg')

    # Add data for storage material tank
    m.data_storage_tank = {
        'LbyD': 0.325,
        'tank_thickness': 0.039,
        'material_density': 7800
    }
    m.fs.charge.l_by_d = pyo.Param(
        initialize=m.data_storage_tank['LbyD'],
        doc='L by D assumption for computing storage tank dimensions')
    m.fs.charge.tank_thickness = pyo.Param(
        initialize=m.data_storage_tank['tank_thickness'],
        doc='Storage tank thickness assumed based on reference')
    m.fs.charge.storage_tank_material_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_material'],
        doc='$/kg of SS316 material')
    m.fs.charge.storage_tank_insulation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_insulation'],
        doc='$/m2')
    m.fs.charge.storage_tank_foundation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_foundation'],
        doc='$/m2')
    m.fs.charge.storage_tank_material_density = pyo.Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3')

    # Add parameters to calculate salt and oil pump costing. Since the
    # pump units are not explicitly modeled, the IDAES cost method is
    # not used for this equipment.  The primary purpose of the salt
    # and oil pump is to move the storage material without changing
    # the pressure. Thus, the pressure head is computed assuming that
    # the salt or oil is moved on an average of 5m linear distance.
    m.data_salt_pump = {
        'FT': 1.5,
        'FM': 2.0,
        'head': 3.281*5,
        'motor_FT': 1,
        'nm': 1
    }
    m.fs.charge.spump_FT = pyo.Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.charge.spump_FM = pyo.Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.charge.spump_head = pyo.Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.charge.spump_motorFT = pyo.Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.charge.spump_nm = pyo.Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')


def _make_constraints(m, add_efficiency=None, power_max=None):
    """Declare the constraints for the charge model
    """

    # Add cooler constraint to ensure that the outlet temperature is
    # subcooled
    @m.fs.charge.cooler.Constraint(m.fs.time,
                                   doc="Outlet temperature to be subcooled")
    def constraint_cooler_enth2(b, t):
        return (
            b.control_volume.properties_out[t].temperature <=
            (b.control_volume.properties_out[t].temperature_sat - 5)
        )

    # Add storage pump pressure constraint
    @m.fs.Constraint(m.fs.time,
                     doc="HX pump out pressure equal to BFP out pressure")
    def constraint_hxpump_presout(b, t):
        return m.fs.charge.hx_pump.outlet.pressure[t] >= \
            (m.main_steam_pressure * 1.1231)

    # Add recycle mixer pressure constraint to ensure the minimum
    # pressure value for outlet pressure
    @m.fs.charge.recycle_mixer.Constraint(m.fs.time,
                                          doc="Recycle mixer outlet pressure \
                                          equal to minimum pressure in inlets")
    def recyclemixer_pressure_constraint(b, t):
        return b.from_bfw_out_state[t].pressure == b.mixed_state[t].pressure

    m.fs.boiler_efficiency = pyo.Var(initialize=0.9,
                                     bounds=(0, 1),
                                     doc="Boiler efficiency")
    m.fs.boiler_efficiency_eq = pyo.Constraint(
        expr=m.fs.boiler_efficiency == (
            0.2143 *
            (m.fs.plant_power_out[0] / power_max) +
            0.7357
        ),
        doc="Boiler efficiency in fraction"
    )
    m.fs.coal_heat_duty = pyo.Var(
        initialize=1000,
        bounds=(0, 1e5),
        doc="Coal heat duty supplied to boiler (MW)")

    if add_efficiency:
        m.fs.coal_heat_duty_eq = pyo.Constraint(
            expr=m.fs.coal_heat_duty *
            m.fs.boiler_efficiency ==
            m.fs.plant_heat_duty[0]
        )
    else:
        m.fs.coal_heat_duty_eq = pyo.Constraint(
            expr=m.fs.coal_heat_duty == m.fs.plant_heat_duty[0]
        )

    m.fs.cycle_efficiency = pyo.Var(initialize=0.4,
                                    bounds=(0, 1),
                                    doc="Cycle efficiency")
    m.fs.cycle_efficiency_eq = pyo.Constraint(
        expr=(
            m.fs.cycle_efficiency *
            m.fs.coal_heat_duty
        ) == m.fs.plant_power_out[0],
        doc="Cycle efficiency"
    )


def disconnect_arcs(m):
    """Disconnect arcs from ultra-supercritical plant base model to
    connect the charge storage system

     """
    for arc_s in [m.fs.boiler_to_turb1,
                  m.fs.bfp_to_fwh8,
                  m.fs.rh1_to_turb3]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def _create_arcs(m):
    """Create arcs to connect the charge storage system to the power plant

    """

    # Disconnect arcs to include charge storage system
    disconnect_arcs(m)

    # Add arcs to connect the charge storage system to the power plant
    m.fs.charge.cooler_to_hxpump = Arc(
        source=m.fs.charge.cooler.outlet,
        destination=m.fs.charge.hx_pump.inlet,
        doc="Connection from cooler to HX pump"
    )
    m.fs.charge.hxpump_to_recyclemix = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer.from_hx_pump,
        doc="Connection from HX pump to recycle mixer"
    )
    m.fs.charge.bfp_to_recyclemix = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.charge.recycle_mixer.from_bfw_out,
        doc="Connection from BFP outlet to recycle mixer"
    )
    m.fs.charge.recyclemix_to_fwh8 = Arc(
        source=m.fs.charge.recycle_mixer.outlet,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from recycle mixer to FWH8 tube side"
    )


def add_disjunction(m):
    """Add disjunction 1 for the storage material selection and
    disjunction 2 for the steam source selection to integrate the
    charge storage system to the power plant model

    """

    # Add disjunction 1 for the storage material selection
    m.fs.salt_disjunction = Disjunction(
        expr=[m.fs.charge.solar_salt_disjunct,
              m.fs.charge.hitec_salt_disjunct,
              m.fs.charge.thermal_oil_disjunct]
    )

    # Add disjunction 2 for the steam source selection
    m.fs.source_disjunction = Disjunction(
        expr=[m.fs.charge.vhp_source_disjunct,
              m.fs.charge.hp_source_disjunct]
    )

    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)

    return m


def solar_salt_disjunct_equations(disj):
    """Block of equations for disjunct 1 in disjunction 1 for the
    selection of Solar salt as the storage material in the charge heat
    exchanger

    """

    m = disj.model()

    # Declare Solar salt heat exchanger unit model
    m.fs.charge.solar_salt_disjunct.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {"property_package": m.fs.prop_water},
            "tube": {"property_package": m.fs.solar_salt_properties}
        }
    )

    solar_hxc = m.fs.charge.solar_salt_disjunct.hxc

    # Calculate heat transfer coefficient for Solar salt heat
    # exchanger
    m.fs.charge.solar_salt_disjunct.hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            (solar_hxc.inlet_2.flow_mass[0] *
             m.fs.charge.hxc_tube_outer_dia) /
            (m.fs.charge.hxc_shell_eff_area *
             solar_hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number")
    m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            solar_hxc.side_2.properties_in[0].cp_specific_heat["Liq"] *
            solar_hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] /
            solar_hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.charge.solar_salt_disjunct.hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            solar_hxc.side_2.properties_out[0].cp_specific_heat["Liq"] *
            solar_hxc.side_2.properties_out[0].dynamic_viscosity["Liq"] /
            solar_hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    m.fs.charge.solar_salt_disjunct.hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 *
            (solar_hxc.salt_reynolds_number**0.6) *
            (solar_hxc.salt_prandtl_number**0.4) *
            ((solar_hxc.salt_prandtl_number /
              solar_hxc.salt_prandtl_wall)**0.25) * (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    m.fs.charge.solar_salt_disjunct.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            solar_hxc.inlet_1.flow_mol[0] *
            solar_hxc.side_1.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia / (
                m.fs.charge.hxc_tube_cs_area *
                m.fs.charge.hxc_n_tubes *
                solar_hxc.side_1.properties_in[0].visc_d_phase["Vap"]
            )
        ),
        doc="Steam Reynolds Number")
    m.fs.charge.solar_salt_disjunct.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (solar_hxc.side_1.properties_in[0].cp_mol /
             solar_hxc.side_1.properties_in[0].mw) *
            solar_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
            solar_hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    m.fs.charge.solar_salt_disjunct.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 *
            (solar_hxc.steam_reynolds_number**0.8) *
            (solar_hxc.steam_prandtl_number**(0.33)) *
            ((solar_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
              solar_hxc.side_1.properties_out[0].visc_d_phase["Liq"])**0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of charge heat exchanger
    m.fs.charge.solar_salt_disjunct.hxc.h_salt = pyo.Expression(
        expr=(
            solar_hxc.side_2.properties_in[0].thermal_conductivity["Liq"] *
            solar_hxc.salt_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient in W/mK")
    m.fs.charge.solar_salt_disjunct.hxc.h_steam = pyo.Expression(
        expr=(
            solar_hxc.side_1.properties_in[0].therm_cond_phase["Vap"] *
            solar_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient in W/mK")

    # Calculate overall heat transfer coefficient for Solar salt
    # charge heat exchanger
    @m.fs.charge.solar_salt_disjunct.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        return (
            solar_hxc.overall_heat_transfer_coefficient[t] * (
                2 * m.fs.charge.hxc_k_steel *
                solar_hxc.h_steam +
                m.fs.charge.hxc_tube_outer_dia *
                m.fs.charge.hxc_log_tube_dia_ratio *
                solar_hxc.h_salt *
                solar_hxc.h_steam +
                m.fs.charge.hxc_tube_dia_ratio *
                solar_hxc.h_salt *
                2 * m.fs.charge.hxc_k_steel)
        ) == (2 * m.fs.charge.hxc_k_steel *
              solar_hxc.h_salt *
              solar_hxc.h_steam)

    # Declare arcs within the disjunct to connect Solar salt heat
    # exchanger to power plant
    m.fs.charge.solar_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=solar_hxc.inlet_1,
        doc="Connection from connector to solar charge heat exchanger"
    )
    m.fs.charge.solar_salt_disjunct.hxc_to_cooler = Arc(
        source=solar_hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet,
        doc="Connection from cooler to solar charge heat exchanger"
    )


def hitec_salt_disjunct_equations(disj):
    """Block of equations for disjunct 2 in disjunction 1 for the
    selection of Hitec salt as the storage material in the charge heat
    exchanger

    """

    m = disj.model()

    # Declare Hitec salt storage heat exchanger
    m.fs.charge.hitec_salt_disjunct.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {"property_package": m.fs.prop_water},
            "tube": {"property_package": m.fs.hitec_salt_properties}
        }
    )

    hitec_hxc = m.fs.charge.hitec_salt_disjunct.hxc

    # Calculate Reynolds, Prandtl, and Nusselt number for the salt and
    # steam side of hitec charge heat exchanger
    m.fs.charge.hitec_salt_disjunct.hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            hitec_hxc.inlet_2.flow_mass[0] *
            m.fs.charge.hxc_tube_outer_dia /
            (m.fs.charge.hxc_shell_eff_area *
             hitec_hxc.side_2.properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            hitec_hxc.side_2.properties_in[0].cp_specific_heat["Liq"] *
            hitec_hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] /
            hitec_hxc.side_2.properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.charge.hitec_salt_disjunct.hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            hitec_hxc.side_2.properties_out[0].cp_specific_heat["Liq"] *
            hitec_hxc.side_2.properties_out[0].dynamic_viscosity["Liq"] /
            hitec_hxc.side_2.properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Wall Prandtl Number"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            1.61 *
            ((hitec_hxc.salt_reynolds_number *
              hitec_hxc.salt_prandtl_number * 0.009)**0.63) *
            ((hitec_hxc.side_2.properties_in[0].dynamic_viscosity["Liq"] /
              hitec_hxc.side_2.properties_out[0].dynamic_viscosity["Liq"])**0.25)
        ),
        doc="Salt Nusslet Number from 2014, He et al, Exp Therm Fl Sci, 59, 9"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            hitec_hxc.inlet_1.flow_mol[0] *
            hitec_hxc.side_1.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia / (
                m.fs.charge.hxc_tube_cs_area *
                m.fs.charge.hxc_n_tubes *
                hitec_hxc.side_1.properties_in[0].visc_d_phase["Vap"]
            )
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (hitec_hxc.side_1.properties_in[0].cp_mol /
             hitec_hxc.side_1.properties_in[0].mw) *
            hitec_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
            hitec_hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 *
            (hitec_hxc.steam_reynolds_number**0.8) *
            (hitec_hxc.steam_prandtl_number**0.33) *
            ((hitec_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
              hitec_hxc.side_1.properties_out[0].visc_d_phase["Liq"])**0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Calculate heat transfer coefficient for salt and steam side of
    # charge heat exchanger
    m.fs.charge.hitec_salt_disjunct.hxc.h_salt = pyo.Expression(
        expr=(
            hitec_hxc.side_2.properties_in[0].thermal_conductivity["Liq"] *
            hitec_hxc.salt_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient in W/mK"
    )
    m.fs.charge.hitec_salt_disjunct.hxc.h_steam = pyo.Expression(
        expr=(
            hitec_hxc.side_1.properties_in[0].therm_cond_phase["Vap"] *
            hitec_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient in W/mK"
    )

    # Calculate overall heat transfer coefficient for Hitec salt heat
    # exchanger
    @m.fs.charge.hitec_salt_disjunct.hxc.Constraint(
        m.fs.time,
        doc="Hitec salt charge heat exchanger \
        overall heat transfer coefficient")
    def constraint_hxc_ohtc_hitec(b, t):
        return (
            hitec_hxc.overall_heat_transfer_coefficient[t] * (
                2 * m.fs.charge.hxc_k_steel *
                hitec_hxc.h_steam +
                m.fs.charge.hxc_tube_outer_dia *
                m.fs.charge.hxc_log_tube_dia_ratio *
                hitec_hxc.h_salt *
                hitec_hxc.h_steam +
                m.fs.charge.hxc_tube_dia_ratio *
                hitec_hxc.h_salt *
                2 * m.fs.charge.hxc_k_steel
            )
        ) == (2 * m.fs.charge.hxc_k_steel *
              hitec_hxc.h_salt * hitec_hxc.h_steam)

    # Declare arcs within the disjunct to connect Hitec salt heat
    # exchanger to power plant
    m.fs.charge.hitec_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=hitec_hxc.inlet_1,
        doc="Connect the connector to Hitec heat exchanger"
    )
    m.fs.charge.hitec_salt_disjunct.hxc_to_cooler = Arc(
        source=hitec_hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet,
        doc="Connect cooler to Hitec charge heat exchanger"
    )


def thermal_oil_disjunct_equations(disj):
    """Block of equations for disjunct 3 in disjunction 1 for the
    selection of thermal oil as the storage material in the charge
    heat exchanger

    """

    m = disj.model()

    # Declare thermal oil heat exchanger
    m.fs.charge.thermal_oil_disjunct.hxc = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {"property_package": m.fs.prop_water},
            "tube": {"property_package": m.fs.therminol66_properties},
            "flow_pattern": HeatExchangerFlowPattern.countercurrent
        }
    )

    thermal_hxc = m.fs.charge.thermal_oil_disjunct.hxc

    # Calculate Reynolds, Prandtl, and Nusselt number for the salt and
    # steam side of thermal oil charge heat exchanger
    m.fs.charge.thermal_oil_disjunct.hxc.oil_in_dynamic_viscosity = pyo.Expression(
        expr=thermal_hxc.side_2.properties_in[0].visc_kin["Liq"] *
        thermal_hxc.side_2.properties_in[0].density["Liq"] * 1e-6
    )

    m.fs.charge.thermal_oil_disjunct.hxc.oil_out_dynamic_viscosity = pyo.Expression(
        expr=thermal_hxc.side_2.properties_out[0].visc_kin["Liq"] *
        thermal_hxc.side_2.properties_out[0].density["Liq"] * 1e-6
    )

    m.fs.charge.thermal_oil_disjunct.hxc.oil_reynolds_number = pyo.Expression(
        expr=(
            thermal_hxc.inlet_2.flow_mass[0] *
            m.fs.charge.hxc_tube_outer_dia /
            (m.fs.charge.hxc_shell_eff_area *
             thermal_hxc.oil_in_dynamic_viscosity)
        ),
        doc="Salt Reynolds Number"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.oil_prandtl_number = pyo.Expression(
        expr=(
            thermal_hxc.side_2.properties_in[0].cp_mass["Liq"] *
            thermal_hxc.oil_in_dynamic_viscosity /
            thermal_hxc.side_2.properties_in[0].therm_cond["Liq"]
        ),
        doc="Salt Prandtl Number")
    m.fs.charge.thermal_oil_disjunct.hxc.oil_prandtl_wall = pyo.Expression(
        expr=(
            thermal_hxc.side_2.properties_out[0].cp_mass["Liq"] *
            thermal_hxc.oil_out_dynamic_viscosity /
            thermal_hxc.side_2.properties_out[0].therm_cond["Liq"]
        ),
        doc="Salt Wall Prandtl Number"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.oil_nusselt_number = pyo.Expression(
        expr=(
            0.36 * ((thermal_hxc.oil_reynolds_number**0.55) *
             (thermal_hxc.oil_prandtl_number**0.33) *
             ((thermal_hxc.oil_prandtl_number /
               thermal_hxc.oil_prandtl_wall)**0.14))
        ),
        doc="Salt Nusslet Number from 2014, He et al, Exp Therm Fl Sci, 59, 9"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            thermal_hxc.inlet_1.flow_mol[0] *
            thermal_hxc.side_1.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia / (
                m.fs.charge.hxc_tube_cs_area *
                m.fs.charge.hxc_n_tubes *
                thermal_hxc.side_1.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (thermal_hxc.side_1.properties_in[0].cp_mol /
             thermal_hxc.side_1.properties_in[0].mw) *
            thermal_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
            thermal_hxc.side_1.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 * (thermal_hxc.steam_reynolds_number**0.8) *
            (thermal_hxc.steam_prandtl_number**(0.33)) * (
                (thermal_hxc.side_1.properties_in[0].visc_d_phase["Vap"] /
                 thermal_hxc.side_1.properties_out[0].visc_d_phase["Liq"])**0.14
            )
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Calculate heat transfer coefficient for salt and steam side of
    # charge heat exchanger
    m.fs.charge.thermal_oil_disjunct.hxc.h_oil = pyo.Expression(
        expr=(
            thermal_hxc.side_2.properties_in[0].therm_cond["Liq"] *
            thermal_hxc.oil_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient in W/mK"
    )
    m.fs.charge.thermal_oil_disjunct.hxc.h_steam = pyo.Expression(
        expr=(
            thermal_hxc.side_1.properties_in[0].therm_cond_phase["Vap"] *
            thermal_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient in W/mK"
    )

    # Calculate overall heat transfer coefficient for thermal oil of
    # thermal oil heat exchanger
    @m.fs.charge.thermal_oil_disjunct.hxc.Constraint(
        m.fs.time,
        doc="Thermal oil charge heat exchanger \
            overall heat transfer coefficient")
    def constraint_hxc_ohtc_thermal_oil(b, t):
        return (
            thermal_hxc.overall_heat_transfer_coefficient[t] * (
                2 * m.fs.charge.hxc_k_steel *
                thermal_hxc.h_steam +
                m.fs.charge.hxc_tube_outer_dia *
                m.fs.charge.hxc_log_tube_dia_ratio *
                thermal_hxc.h_oil *
                thermal_hxc.h_steam +
                m.fs.charge.hxc_tube_dia_ratio *
                thermal_hxc.h_oil *
                2 * m.fs.charge.hxc_k_steel
            )
        ) == (2 * m.fs.charge.hxc_k_steel *
              thermal_hxc.h_oil *
              thermal_hxc.h_steam)

    # Declare arcs within the disjunct to connect thermal oil heat
    # exchanger to power plant
    m.fs.charge.thermal_oil_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=thermal_hxc.inlet_1,
        doc="Connection from connector to thermal oil charge heat exchanger"
    )
    m.fs.charge.thermal_oil_disjunct.hxc_to_cooler = Arc(
        source=thermal_hxc.outlet_1,
        destination=m.fs.charge.cooler.inlet,
        doc="Connection from cooler to thermal oil charge heat exchanger"
    )


def vhp_source_disjunct_equations(disj):
    """Block of equations for disjunct 1 in disjunction 2 for the
    selection of a very high-pressure steam source to heat up the
    storage material in charge heat exchanger

    """

    m = disj.model()

    # Add splitter to send a portion of steam to the charge storage
    # system
    m.fs.charge.vhp_source_disjunct.ess_vhp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxc", "to_turbine"],
        }
    )

    # Define arc to connect boiler to vhp splitter
    m.fs.charge.vhp_source_disjunct.boiler_to_essvhp = Arc(
        source=m.fs.boiler.outlet,
        destination=m.fs.charge.vhp_source_disjunct.ess_vhp_split.inlet,
        doc="Connection from boiler to hp splitter"
    )

    # Define arc to connect vhp splitter to turbine 1
    m.fs.charge.vhp_source_disjunct.essvhp_to_turb1 = Arc(
        source=m.fs.charge.vhp_source_disjunct.ess_vhp_split.to_turbine,
        destination=m.fs.turbine[1].inlet,
        doc="Connection from VHP splitter to turbine 1"
    )

    # Define arc to connect vhp splitter to connector
    m.fs.charge.vhp_source_disjunct.vhpsplit_to_connector = Arc(
        source=m.fs.charge.vhp_source_disjunct.ess_vhp_split.to_hxc,
        destination=m.fs.charge.connector.inlet,
        doc="Connection from VHP splitter to connector"
    )

    # Define arc to re-connect reheater 1 to turbine 3
    m.fs.charge.vhp_source_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from reheater 1 to turbine 3"
    )


def hp_source_disjunct_equations(disj):
    """Block of equations for disjunct 2 in disjunction 2 for the
    selection of a high-pressure steam source to heat up the storage
    material in charge heat exchanger

    """

    m = disj.model()

    # Add splitter to send a portion of steam to the charge storage
    # system
    m.fs.charge.hp_source_disjunct.ess_hp_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_hxc", "to_turbine"],
        }
    )

    # Define arcs to connect reheater 1 to hp splitter
    m.fs.charge.hp_source_disjunct.rh1_to_esshp = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.charge.hp_source_disjunct.ess_hp_split.inlet,
        doc="Connection from reheater to ip splitter"
    )

    # Define arcs to connect hp splitter to turbine 3
    m.fs.charge.hp_source_disjunct.esshp_to_turb3 = Arc(
        source=m.fs.charge.hp_source_disjunct.ess_hp_split.to_turbine,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from HP splitter to turbine 3"
    )

    # Define arcs to connect hp splitter to cooler
    m.fs.charge.hp_source_disjunct.hpsplit_to_connector = Arc(
        source=m.fs.charge.hp_source_disjunct.ess_hp_split.to_hxc,
        destination=m.fs.charge.connector.inlet,
        doc="Connection from HP splitter to connector"
    )

    # Define arc to connect boiler to turbine 1
    m.fs.charge.hp_source_disjunct.boiler_to_turb1 = Arc(
        source=m.fs.boiler.outlet,
        destination=m.fs.turbine[1].inlet,
        doc="Connection from VHP splitter to turbine 1"
    )


def set_model_input(m):
    """Define model inputs such as fixed variables and parameter
    values. The parameter values in this block, unless otherwise
    stated explicitly, are either assumed or estimated for a total
    power out of 437 MW. The inputs fixed in this function are the
    necessary inputs to obtain a square model (0 degrees of freedom).

    Unless stated otherwise, the units are: temperature in K, pressure
    in Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    in W

    """


    ###########################################################################
    # Fix data in charge system
    ###########################################################################
    # Add heat exchanger area from supercritical plant model_input. For
    # conceptual design optimization, area is unfixed and optimized
    m.fs.charge.solar_salt_disjunct.hxc.area.fix(100)
    m.fs.charge.hitec_salt_disjunct.hxc.area.fix(100)
    m.fs.charge.thermal_oil_disjunct.hxc.area.fix(2500)

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass.fix(100)
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.temperature.fix(513.15)
    m.fs.charge.solar_salt_disjunct.hxc.inlet_2.pressure.fix(101325)

    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass.fix(100)
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.temperature.fix(435.15)
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.pressure.fix(101325)

    m.fs.charge.thermal_oil_disjunct.hxc.inlet_2.flow_mass[0].fix(700)
    m.fs.charge.thermal_oil_disjunct.hxc.inlet_2.temperature[0].fix(353.15)
    m.fs.charge.thermal_oil_disjunct.hxc.inlet_2.pressure[0].fix(101325)

    # Cooler outlet enthalpy is fixed during model build to ensure the
    # inlet to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler
    # is constrained in the model
    m.fs.charge.cooler.outlet.enth_mol[0].fix(10000)
    m.fs.charge.cooler.deltaP[0].fix(0)

    # HX pump efficiecncy assumption
    m.fs.charge.hx_pump.efficiency_pump.fix(0.80)
    m.fs.charge.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    ###########################################################################
    # Fix data in steam source splitters
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.charge.vhp_source_disjunct.ess_vhp_split.split_fraction[0, "to_hxc"].fix(0.01)
    m.fs.charge.hp_source_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.01)

    ###########################################################################
    # Fix data in connector
    ###########################################################################
    # Fix heat duty to zero for dummy conector
    m.fs.charge.connector.heat_duty[0].fix(0)


def set_scaling_factors(m):
    """Set scaling factors in the flowsheet

    """

    # Include scaling factors for solar, hitec, and thermal oil charge
    # heat exchangers
    for fluid in [m.fs.charge.solar_salt_disjunct.hxc,
                  m.fs.charge.hitec_salt_disjunct.hxc,
                  m.fs.charge.thermal_oil_disjunct.hxc]:
        iscale.set_scaling_factor(fluid.area, 1e-2)
        iscale.set_scaling_factor(
            fluid.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(fluid.shell.heat, 1e-6)
        iscale.set_scaling_factor(fluid.tube.heat, 1e-6)

    iscale.set_scaling_factor(m.fs.charge.hx_pump.control_volume.work, 1e-6)

    for k in [m.fs.charge.cooler, m.fs.charge.connector]:
        iscale.set_scaling_factor(k.control_volume.heat, 1e-6)


def initialize(m, solver=None, outlvl=idaeslog.NOTSET):
    """Initialize the units included in the charge model

    """

    print()
    # Add options to NLP solver
    optarg = {"max_iter": 300,
              "tol": 1e-8,
              "halt_on_ampl_error": "yes"}
    solver = get_solver(solver, optarg)

    # Include scaling factors
    iscale.calculate_scaling_factors(m)

    # Initialize splitters
    propagate_state(m.fs.charge.vhp_source_disjunct.boiler_to_essvhp)
    m.fs.charge.vhp_source_disjunct.ess_vhp_split.initialize(outlvl=outlvl,
                                                             optarg=solver.options)
    propagate_state(m.fs.charge.hp_source_disjunct.rh1_to_esshp)
    m.fs.charge.hp_source_disjunct.ess_hp_split.initialize(outlvl=outlvl,
                                                           optarg=solver.options)

    # Reinitialize turbines connected to splitters since a portion of
    # the flow is now sent to the charge storage system
    propagate_state(m.fs.charge.hp_source_disjunct.boiler_to_turb1)
    m.fs.turbine[1].inlet.fix()
    m.fs.turbine[1].initialize(outlvl=outlvl,
                               optarg=solver.options)
    propagate_state(m.fs.charge.hp_source_disjunct.esshp_to_turb3)
    m.fs.turbine[3].inlet.fix()
    m.fs.turbine[3].initialize(outlvl=outlvl,
                               optarg=solver.options)

    # Initialize connector
    propagate_state(m.fs.charge.hp_source_disjunct.hpsplit_to_connector)
    m.fs.charge.connector.inlet.fix()
    m.fs.charge.connector.initialize(outlvl=outlvl,
                                     optarg=solver.options)

    # Initialize Solar salt, Hitec salt, and thermal oil storage heat
    # exchanger. Fix the charge steam inlet during initialization and
    # unfix during optimization
    propagate_state(m.fs.charge.solar_salt_disjunct.connector_to_hxc)
    m.fs.charge.solar_salt_disjunct.hxc.inlet_1.fix()
    m.fs.charge.solar_salt_disjunct.hxc.initialize(outlvl=outlvl,
                                                   optarg=solver.options)

    propagate_state(m.fs.charge.hitec_salt_disjunct.connector_to_hxc)
    m.fs.charge.hitec_salt_disjunct.hxc.inlet_1.fix()
    m.fs.charge.hitec_salt_disjunct.hxc.initialize(
        outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.charge.thermal_oil_disjunct.connector_to_hxc)
    m.fs.charge.thermal_oil_disjunct.hxc.inlet_1.fix()
    m.fs.charge.thermal_oil_disjunct.hxc.initialize(outlvl=outlvl)

    # Initialize cooler
    propagate_state(m.fs.charge.solar_salt_disjunct.hxc_to_cooler)
    m.fs.charge.cooler.inlet.fix()
    m.fs.charge.cooler.initialize(outlvl=outlvl,
                                  optarg=solver.options)

    # Initialize HX pump
    propagate_state(m.fs.charge.cooler_to_hxpump)
    m.fs.charge.hx_pump.initialize(outlvl=outlvl,
                                   optarg=solver.options)

    # Initialize recycle mixer
    propagate_state(m.fs.charge.bfp_to_recyclemix)
    propagate_state(m.fs.charge.hxpump_to_recyclemix)
    m.fs.charge.recycle_mixer.initialize(outlvl=outlvl)

    # Assert 0 degrees of freedom to solve a square problem
    assert degrees_of_freedom(m) == 0

    # Solve units initialization
    init_results = solver.solve(m, options=optarg)
    print("Charge model initialization solver termination = ",
          init_results.solver.termination_condition)
    print("***************   Charge Model Initialized   ********************")


def build_costing(m, solver=None):
    """Add cost correlations for the storage design analysis

    This function is used to estimate the capital and operatig cost of
    integrating a charge storage system to the power plant and it
    contains cost correlations to estimate: (i) annualized capital
    cost of charge heat exchanger, salt storage tank, molten salt
    pump, and salt inventory, and (ii) the operating costs for 1
    year. Unless otherwise stated, the cost correlations used here,
    except for IDAES costing method, are taken from 2nd Edition,
    Product & Process Design Principles, Seider et al.

    """

    # Add options to NLP solver
    optarg = {"tol": 1e-8,
              "max_iter": 300}

    ###########################################################################
    # Add capital cost
    # 1. Calculate charge storage material purchase cost
    # 2. Calculate charge storage heat exchangers cost
    # 3. Calculate charge storage material pump purchase cost
    # 4. Calculate charge storage material vessel cost
    # 5. Calculate total capital cost for charge storage system
    ###########################################################################
    # Add capital cost: 1. Calculate storage material purchase cost
    ###########################################################################

    # Solar salt inventory
    m.fs.charge.solar_salt_disjunct.salt_amount = pyo.Expression(
        expr=(m.fs.charge.solar_salt_disjunct.hxc.inlet_2.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total amount of Solar salt in kg"
    )
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Purchase cost of Solar salt in $ per year"
    )

    def solar_salt_purchase_cost_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.salt_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.solar_salt_disjunct.salt_amount *
                m.fs.charge.solar_salt_price
            )
        )
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost_eq = pyo.Constraint(
        rule=solar_salt_purchase_cost_rule)

    # Hitec salt inventory
    m.fs.charge.hitec_salt_disjunct.salt_amount = pyo.Expression(
        expr=(m.fs.charge.hitec_salt_disjunct.hxc.inlet_2.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total amount of Hitec salt in kg"
    )
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Purchase cost of Hitec salt in $ per year"
    )

    def hitec_salt_purchase_cost_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.salt_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.hitec_salt_disjunct.salt_amount *
                m.fs.charge.hitec_salt_price
            )
        )
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost_eq = pyo.Constraint(
        rule=hitec_salt_purchase_cost_rule)

    # Thermal oil inventory
    m.fs.charge.thermal_oil_disjunct.oil_amount = pyo.Expression(
        expr=(m.fs.charge.thermal_oil_disjunct.hxc.inlet_2.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total amount of thermal oil in kg"
    )
    m.fs.charge.thermal_oil_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e10),
        doc="Purchase cost of thermal oil in $ per year"
    )

    def thermal_oil_purchase_cost_rule(b):
        return (
            m.fs.charge.thermal_oil_disjunct.salt_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.thermal_oil_disjunct.oil_amount *
                m.fs.charge.thermal_oil_price
            )
        )
    m.fs.charge.thermal_oil_disjunct.salt_purchase_cost_eq = pyo.Constraint(
        rule=thermal_oil_purchase_cost_rule)

    # Initialize Solar, Hitec, and thermal oil purchase cost variables
    for salt_disj in [m.fs.charge.solar_salt_disjunct,
                      m.fs.charge.hitec_salt_disjunct,
                      m.fs.charge.thermal_oil_disjunct]:
        calculate_variable_from_constraint(
            salt_disj.salt_purchase_cost,
            salt_disj.salt_purchase_cost_eq)

    ###########################################################################
    # Add capital cost: 2. Calculate charge storage heat exchangers cost
    ###########################################################################

    # Calculate and initialize Solar salt, Hitec salt, and thermal oil
    # charge heat exchangers costs, which are estimated using the
    # IDAES costing method with default options, i.e. a U-tube heat
    # exchanger, stainless steel material, and a tube length of
    # 12ft. Refer to costing documentation to change any of the
    # default options Purchase cost of heat exchanger has to be
    # annualized when used
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc,
                     m.fs.charge.thermal_oil_disjunct.hxc]:
        salt_hxc.get_costing()
        salt_hxc.costing.CE_index = m.CE_index
        icost.initialize(salt_hxc.costing)

    # Calculate and initialize storage water pump cost. The purchase
    # cost has to be annualized when used
    m.fs.charge.hx_pump.get_costing(
        Mat_factor="stain_steel",
        mover_type="compressor",
        compressor_type="centrifugal",
        driver_mover_type="electrical_motor",
        pump_type="centrifugal",
        pump_type_factor='1.4',
        pump_motor_type_factor='open'
        )
    m.fs.charge.hx_pump.costing.CE_index = m.CE_index
    icost.initialize(m.fs.charge.hx_pump.costing)

    ###########################################################################
    # Add capital cost: 3. Calculate charge storage material pump purchase cost
    ###########################################################################

    # ---------- Solar salt ----------

    # Calculate purchase cost of Solar salt pump
    m.fs.charge.solar_salt_disjunct.spump_Qgpm = pyo.Expression(
        expr=(
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].flow_mass *
            264.17 * 60 /
            (m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"])
        ),
        doc="Conversion of Solar salt flow mass to volumetric flow in gallons per min"
    )
    m.fs.charge.solar_salt_disjunct.dens_lbft3 = pyo.Expression(
        expr=m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] *
        0.062428
    )
    m.fs.charge.solar_salt_disjunct.spump_sf = pyo.Expression(
        expr=(m.fs.charge.solar_salt_disjunct.spump_Qgpm *
              (m.fs.charge.spump_head**0.5)),
        doc="Pump size factor"
    )
    m.fs.charge.solar_salt_disjunct.pump_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_FT * m.fs.charge.spump_FM *
            exp(9.2951 -
                0.6019 * log(m.fs.charge.solar_salt_disjunct.spump_sf) +
                0.0519 * ((log(m.fs.charge.solar_salt_disjunct.spump_sf))**2))
        ),
        doc="Base purchase cost of Solar salt pump in $"
    )

    # Calculate cost of Solar salt pump motor
    m.fs.charge.solar_salt_disjunct.spump_np = pyo.Expression(
        expr=(
            -0.316 +
            0.24015 * log(m.fs.charge.solar_salt_disjunct.spump_Qgpm) -
            0.01199 * ((log(m.fs.charge.solar_salt_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump in horsepower"
    )
    m.fs.charge.solar_salt_disjunct.motor_pc = pyo.Expression(
        expr=(
            (m.fs.charge.solar_salt_disjunct.spump_Qgpm *
             m.fs.charge.spump_head *
             m.fs.charge.solar_salt_disjunct.dens_lbft3) /
            (33000 * m.fs.charge.solar_salt_disjunct.spump_np *
             m.fs.charge.spump_nm)
        ),
        doc="Power consumption of motor in horsepower"
    )

    log_motor_pc = log(m.fs.charge.solar_salt_disjunct.motor_pc)
    m.fs.charge.solar_salt_disjunct.motor_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(
                5.4866 +
                0.13141 * log_motor_pc +
                0.053255 * (log_motor_pc**2) +
                0.028628 * (log_motor_pc**3) -
                0.0035549 * (log_motor_pc**4)
            )
        ),
        doc="Base cost of Solar Salt pump's motor in $"
    )

    # Calculate and initialize the total cost of Solar salt pump
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Total purchase cost of Solar salt pump in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.spump_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.solar_salt_disjunct.pump_CP +
                m.fs.charge.solar_salt_disjunct.motor_CP) *
            (m.CE_index / 394)
        )
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost,
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq)

    
    # ---------- Hitec salt ----------

    # Calculate cost of Hitec salt pump
    m.fs.charge.hitec_salt_disjunct.spump_Qgpm = pyo.Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].flow_mass *
            264.17 * 60 /
            (m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"])
        ),
        doc="Conversion of Hitec salt flow mass to volumetric flow in gallons per min"
    )
    m.fs.charge.hitec_salt_disjunct.dens_lbft3 = pyo.Expression(
        expr=m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] *
        0.062428
    )
    m.fs.charge.hitec_salt_disjunct.spump_sf = pyo.Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.spump_Qgpm *
            (m.fs.charge.spump_head**0.5)
        ),
        doc="Pump size factor"
    )

    log_hitec_spump_sf = log(m.fs.charge.hitec_salt_disjunct.spump_sf)
    m.fs.charge.hitec_salt_disjunct.pump_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_FT *
            m.fs.charge.spump_FM *
            exp(
                9.2951 -
                0.6019 * log_hitec_spump_sf +
                0.0519 * (log_hitec_spump_sf**2)
            )
        ),
        doc="Base purchase cost of Hitec salt pump in $"
    )

    # Calculate cost of Hitec salt pump motor
    m.fs.charge.hitec_salt_disjunct.spump_np = pyo.Expression(
        expr=(
            -0.316 +
            0.24015 * log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm) -
            0.01199 * ((log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump in horsepower"
    )
    m.fs.charge.hitec_salt_disjunct.motor_pc = pyo.Expression(
        expr=(
            (m.fs.charge.hitec_salt_disjunct.spump_Qgpm *
             m.fs.charge.spump_head *
             m.fs.charge.hitec_salt_disjunct.dens_lbft3) /
            (33000 *
             m.fs.charge.hitec_salt_disjunct.spump_np *
             m.fs.charge.spump_nm)
        ),
        doc="Power consumption of motor in horsepower"
    )

    log_hitec_motor_pc = log(m.fs.charge.hitec_salt_disjunct.motor_pc)
    m.fs.charge.hitec_salt_disjunct.motor_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(
                5.4866 +
                0.13141 * log_hitec_motor_pc +
                0.053255 * (log_hitec_motor_pc**2) +
                0.028628 * (log_hitec_motor_pc**3) -
                0.0035549 * (log_hitec_motor_pc**4))
        ),
        doc="Base cost of Hitec salt pump's motor in $"
    )

    # Calculate and initialize total purchase cost of Hitec salt pump
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Total purchase cost of Hitec salt pump in $"
    )

    def hitec_spump_purchase_cost_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.spump_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.hitec_salt_disjunct.pump_CP +
                m.fs.charge.hitec_salt_disjunct.motor_CP) *
            (m.CE_index / 394)
        )
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq = pyo.Constraint(
        rule=hitec_spump_purchase_cost_rule)

    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost,
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq)


    # ---------- Thermal oil ----------

    # Calculate cost of thermal oil pump
    m.fs.charge.thermal_oil_disjunct.spump_Qgpm = pyo.Expression(
        expr=(
            m.fs.charge.thermal_oil_disjunct.hxc.side_2.properties_in[0].flow_mass *
            264.17 * 60 /
            (m.fs.charge.thermal_oil_disjunct.hxc.side_2.properties_in[0].density["Liq"])
        ),
        doc="Conversion of thermal oil flow mass to volumetric flow in gallons per min"
    )
    m.fs.charge.thermal_oil_disjunct.dens_lbft3 = pyo.Expression(
        expr=m.fs.charge.thermal_oil_disjunct.hxc.side_2.properties_in[0].density["Liq"] *
        0.062428
    )
    m.fs.charge.thermal_oil_disjunct.spump_sf = pyo.Expression(
        expr=(m.fs.charge.thermal_oil_disjunct.spump_Qgpm *
              (m.fs.charge.spump_head**0.5)),
        doc="Pump size factor"
    )

    log_thermal_oil_spump_sf = log(m.fs.charge.thermal_oil_disjunct.spump_sf)
    m.fs.charge.thermal_oil_disjunct.pump_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_FT * m.fs.charge.spump_FM *
            exp(
                9.2951 -
                0.6019 * log_thermal_oil_spump_sf +
                0.0519 * (log_thermal_oil_spump_sf**2)
            )
        ),
        doc="Base purchase cost of thermal oil pump in $"
    )

    # Calculate cost of thermal oil pump motor
    m.fs.charge.thermal_oil_disjunct.spump_np = pyo.Expression(
        expr=(
            -0.316 +
            0.24015 * log(m.fs.charge.thermal_oil_disjunct.spump_Qgpm) -
            0.01199 * ((log(m.fs.charge.thermal_oil_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump in horsepower"
    )
    m.fs.charge.thermal_oil_disjunct.motor_pc = pyo.Expression(
        expr=(
            (m.fs.charge.thermal_oil_disjunct.spump_Qgpm *
             m.fs.charge.spump_head *
             m.fs.charge.thermal_oil_disjunct.dens_lbft3) /
            (33000 *
             m.fs.charge.thermal_oil_disjunct.spump_np *
             m.fs.charge.spump_nm)
        ),
        doc="Power consumption of motor in horsepower"
    )

    log_thermal_oil_motor_pc = log(m.fs.charge.thermal_oil_disjunct.motor_pc)
    m.fs.charge.thermal_oil_disjunct.motor_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(
                5.4866 +
                0.13141 * log_thermal_oil_motor_pc +
                0.053255 * (log_thermal_oil_motor_pc**2) +
                0.028628 * (log_thermal_oil_motor_pc**3) -
                0.0035549 * (log_thermal_oil_motor_pc**4)
            )
        ),
        doc="Base cost of thermal oil pump's motor in $"
    )

    # Calculate and initialize total purchase cost of thermal oil total pump
    m.fs.charge.thermal_oil_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e10),
        doc="Total purchase cost of thermal oil pump in $"
    )

    def oil_spump_purchase_cost_rule(b):
        return (
            m.fs.charge.thermal_oil_disjunct.spump_purchase_cost *
            m.fs.charge.num_of_years == (
                m.fs.charge.thermal_oil_disjunct.pump_CP +
                m.fs.charge.thermal_oil_disjunct.motor_CP) *
            (m.CE_index / 394)
        )
    m.fs.charge.thermal_oil_disjunct.spump_purchase_cost_eq = pyo.Constraint(
        rule=oil_spump_purchase_cost_rule)

    calculate_variable_from_constraint(
        m.fs.charge.thermal_oil_disjunct.spump_purchase_cost,
        m.fs.charge.thermal_oil_disjunct.spump_purchase_cost_eq)

    ###########################################################################
    # Add capital cost: 4. Calculate charge storage material vertical vessel cost
    ###########################################################################

    # ---------- Solar salt ----------

    # Calculate size and dimensions of Solar salt storage tank
    m.fs.charge.solar_salt_disjunct.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank with 20% excess capacity")
    m.fs.charge.solar_salt_disjunct.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.solar_salt_disjunct.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.solar_salt_disjunct.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank in m")
    m.fs.charge.solar_salt_disjunct.no_of_tanks = pyo.Var(
        initialize=1,
        bounds=(1, 3),
        doc='No of Tank units to use cost correlations')

    # Fix the number of Solar salt storage tanks
    m.fs.charge.solar_salt_disjunct.no_of_tanks.fix()

    # Compute Solar salt tank volume with a 10% margin
    def solar_tank_volume_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_volume *
            m.fs.charge.solar_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] ==
            m.fs.charge.solar_salt_disjunct.salt_amount *
            1.10
        )
    m.fs.charge.solar_salt_disjunct.tank_volume_eq = pyo.Constraint(
        rule=solar_tank_volume_rule)

    # Compute Solar salt tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def solar_tank_surf_area_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_surf_area == (
                pi * m.fs.charge.solar_salt_disjunct.tank_diameter *
                m.fs.charge.solar_salt_disjunct.tank_height) +
            (pi * m.fs.charge.solar_salt_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge.solar_salt_disjunct.tank_surf_area_eq = pyo.Constraint(
        rule=solar_tank_surf_area_rule)

    # Compute Solar salt tank diameter for an assumed lenght and
    # diameter
    def solar_tank_diameter_rule(b):
        return (
            m.fs.charge.solar_salt_disjunct.tank_diameter == (
                (4 * (m.fs.charge.solar_salt_disjunct.tank_volume /
                      m.fs.charge.solar_salt_disjunct.no_of_tanks) /
                 (m.fs.charge.l_by_d * pi))**(1 / 3))
        )
    m.fs.charge.solar_salt_disjunct.tank_diameter_eq = pyo.Constraint(
        rule=solar_tank_diameter_rule)

    # Compute height of Solar salt tank
    def solar_tank_height_rule(b):
        return m.fs.charge.solar_salt_disjunct.tank_height == (
            m.fs.charge.l_by_d *
            m.fs.charge.solar_salt_disjunct.tank_diameter)
    m.fs.charge.solar_salt_disjunct.tank_height_eq = pyo.Constraint(
        rule=solar_tank_height_rule)

    # Initialize Solar salt tank design correlations
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_volume,
        m.fs.charge.solar_salt_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_diameter,
        m.fs.charge.solar_salt_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.tank_height,
        m.fs.charge.solar_salt_disjunct.tank_height_eq)

    # Declare a dummy pyomo block for Solar salt storage tank for
    # costing
    m.fs.charge.solar_salt_disjunct.costing = pyo.Block()

    m.fs.charge.solar_salt_disjunct.costing.tank_material_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_tank_material_cost(b):
        return m.fs.charge.solar_salt_disjunct.costing.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_density *
            m.fs.charge.solar_salt_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_tank_material_cost)

    def rule_tank_insulation_cost(b):
        return (
            m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost == (
                m.fs.charge.storage_tank_insulation_cost *
                m.fs.charge.solar_salt_disjunct.tank_surf_area))

    m.fs.charge.solar_salt_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_tank_insulation_cost)

    def rule_tank_foundation_cost(b):
        return (
            m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost == (
                m.fs.charge.storage_tank_foundation_cost *
                pi *
                m.fs.charge.solar_salt_disjunct.tank_diameter**2 / 4)
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge.solar_salt_disjunct.costing.total_tank_cost = pyo.Expression(
        expr=m.fs.charge.solar_salt_disjunct.costing.tank_material_cost +
        m.fs.charge.solar_salt_disjunct.costing.tank_foundation_cost +
        m.fs.charge.solar_salt_disjunct.costing.tank_insulation_cost
    )


    # ---------- Hitec salt ----------

    # Calculate size and dimension of Hitec salt storage tank
    m.fs.charge.hitec_salt_disjunct.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 10000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank with 20% excess capacity")
    m.fs.charge.hitec_salt_disjunct.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 5000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.hitec_salt_disjunct.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.hitec_salt_disjunct.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank in m")
    m.fs.charge.hitec_salt_disjunct.no_of_tanks = pyo.Var(
        initialize=1,
        bounds=(1, 4),
        doc='No of Tank units to use cost correlations')

    # Fix the number of Hitec salt storage tanks
    m.fs.charge.hitec_salt_disjunct.no_of_tanks.fix()

    # Compute Hitec salt tank volume with a 10% margin
    def hitec_tank_volume_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.tank_volume *
            m.fs.charge.hitec_salt_disjunct.hxc.side_2.properties_in[0].density["Liq"] ==
            m.fs.charge.hitec_salt_disjunct.salt_amount *
            1.10
        )
    m.fs.charge.hitec_salt_disjunct.tank_volume_eq = pyo.Constraint(
        rule=hitec_tank_volume_rule)

    # Compute Hitec salt tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def hitec_tank_surf_area_rule(b):
        return m.fs.charge.hitec_salt_disjunct.tank_surf_area == (
            (pi * m.fs.charge.hitec_salt_disjunct.tank_diameter *
             m.fs.charge.hitec_salt_disjunct.tank_height) +
            (pi *
             m.fs.charge.hitec_salt_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge.hitec_salt_disjunct.tank_surf_area_eq = pyo.Constraint(
        rule=hitec_tank_surf_area_rule)

    # Compute Hitec salt tank diameter for an assumed lenght and diameter
    def hitec_tank_diameter_rule(b):
        return (
            m.fs.charge.hitec_salt_disjunct.tank_diameter == (
                (4 * (m.fs.charge.hitec_salt_disjunct.tank_volume /
                      m.fs.charge.hitec_salt_disjunct.no_of_tanks) /
                 (m.fs.charge.l_by_d * pi))**(1 / 3))
        )
    m.fs.charge.hitec_salt_disjunct.tank_diameter_eq = pyo.Constraint(
        rule=hitec_tank_diameter_rule)

    # Compute Hitec salt tank height
    def hitec_tank_height_rule(b):
        return m.fs.charge.hitec_salt_disjunct.tank_height == (
            m.fs.charge.l_by_d *
            m.fs.charge.hitec_salt_disjunct.tank_diameter
        )
    m.fs.charge.hitec_salt_disjunct.tank_height_eq = pyo.Constraint(
        rule=hitec_tank_height_rule)

    # Initialize Hitec salt tank design correlations
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_volume,
        m.fs.charge.hitec_salt_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_diameter,
        m.fs.charge.hitec_salt_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.tank_height,
        m.fs.charge.hitec_salt_disjunct.tank_height_eq)

    # Declare a dummy pyomo block for Hitec salt storage tank for
    # costing
    m.fs.charge.hitec_salt_disjunct.costing = Block()

    m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_hitec_tank_material_cost(b):
        return m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_density *
            m.fs.charge.hitec_salt_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_hitec_tank_material_cost)

    def rule_hitec_tank_insulation_cost(b):
        return (m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost ==
                m.fs.charge.storage_tank_insulation_cost *
                m.fs.charge.hitec_salt_disjunct.tank_surf_area)
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_hitec_tank_insulation_cost)

    def rule_hitec_tank_foundation_cost(b):
        return (m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost ==
                m.fs.charge.storage_tank_foundation_cost *
                pi *
                m.fs.charge.hitec_salt_disjunct.tank_diameter**2 / 4)
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_hitec_tank_foundation_cost)

    # Add expression to compute the total cost for the Hitec salt tank
    m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost = pyo.Expression(
        expr=m.fs.charge.hitec_salt_disjunct.costing.tank_material_cost +
        m.fs.charge.hitec_salt_disjunct.costing.tank_foundation_cost +
        m.fs.charge.hitec_salt_disjunct.costing.tank_insulation_cost
    )

    # ---------- Thermal oil ----------

    # Calculate size and dimension of thermal oil storage tank
    m.fs.charge.thermal_oil_disjunct.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 20000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank with 20% excess capacity")
    m.fs.charge.thermal_oil_disjunct.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 6000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.thermal_oil_disjunct.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 41),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.thermal_oil_disjunct.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 14),
        units=pyunits.m,
        doc="Length of the salt tank in m")
    m.fs.charge.thermal_oil_disjunct.no_of_tanks = pyo.Var(
        initialize=1,
        bounds=(1, 4),
        doc='No of Tank units to use cost correlations')

    # Fix the number of thermal oil tanks
    m.fs.charge.thermal_oil_disjunct.no_of_tanks.fix()

    # Compute thermal oil tank volume with a 10% margin
    def oil_tank_volume_rule(b):
        return (
            m.fs.charge.thermal_oil_disjunct.tank_volume *
            m.fs.charge.thermal_oil_disjunct.hxc.
            side_2.properties_in[0].density["Liq"] ==
            m.fs.charge.thermal_oil_disjunct.oil_amount *
            1.10
        )
    m.fs.charge.thermal_oil_disjunct.tank_volume_eq = pyo.Constraint(
        rule=oil_tank_volume_rule)

    # Compute thermal oil tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def oil_tank_surf_area_rule(b):
        return m.fs.charge.thermal_oil_disjunct.tank_surf_area == (
            (pi *
             m.fs.charge.thermal_oil_disjunct.tank_diameter *
             m.fs.charge.thermal_oil_disjunct.tank_height) +
            (pi *
             m.fs.charge.thermal_oil_disjunct.tank_diameter**2) / 4
        )
    m.fs.charge.thermal_oil_disjunct.tank_surf_area_eq = pyo.Constraint(
        rule=oil_tank_surf_area_rule)

    # Compute thermal oil tank diameter for an assumed lenght and diameter
    def oil_tank_diameter_rule(b):
        return (
            m.fs.charge.thermal_oil_disjunct.tank_diameter == (
                (4 * (m.fs.charge.thermal_oil_disjunct.tank_volume /
                      m.fs.charge.thermal_oil_disjunct.no_of_tanks) /
                 (m.fs.charge.l_by_d * pi))**(1 / 3))
        )
    m.fs.charge.thermal_oil_disjunct.tank_diameter_eq = pyo.Constraint(
        rule=oil_tank_diameter_rule)

    # Compute height of thermal oil tank
    def oil_tank_height_rule(b):
        return m.fs.charge.thermal_oil_disjunct.tank_height == (
            m.fs.charge.l_by_d *
            m.fs.charge.thermal_oil_disjunct.tank_diameter
        )
    m.fs.charge.thermal_oil_disjunct.tank_height_eq = pyo.Constraint(
        rule=oil_tank_height_rule)

    # Initialize thermal oil tank design correlations
    calculate_variable_from_constraint(
        m.fs.charge.thermal_oil_disjunct.tank_volume,
        m.fs.charge.thermal_oil_disjunct.tank_volume_eq)
    calculate_variable_from_constraint(
        m.fs.charge.thermal_oil_disjunct.tank_diameter,
        m.fs.charge.thermal_oil_disjunct.tank_diameter_eq)
    calculate_variable_from_constraint(
        m.fs.charge.thermal_oil_disjunct.tank_height,
        m.fs.charge.thermal_oil_disjunct.tank_height_eq)

    # Declare a dummy pyomo block for thermal oil storage tank for
    # costing
    m.fs.charge.thermal_oil_disjunct.costing = Block()

    m.fs.charge.thermal_oil_disjunct.costing.tank_material_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.thermal_oil_disjunct.costing.tank_insulation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )
    m.fs.charge.thermal_oil_disjunct.costing.tank_foundation_cost = pyo.Var(
        initialize=5000,
        bounds=(1000, 1e7)
    )

    def rule_oil_tank_material_cost(b):
        return m.fs.charge.thermal_oil_disjunct.costing.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_density *
            m.fs.charge.thermal_oil_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_oil_tank_material_cost)

    def rule_oil_tank_insulation_cost(b):
        return (
            m.fs.charge.thermal_oil_disjunct.costing.tank_insulation_cost ==
            m.fs.charge.storage_tank_insulation_cost *
            m.fs.charge.thermal_oil_disjunct.tank_surf_area)
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_oil_tank_insulation_cost)

    def rule_oil_tank_foundation_cost(b):
        return (
            m.fs.charge.thermal_oil_disjunct.costing.tank_foundation_cost ==
            m.fs.charge.storage_tank_foundation_cost *
            pi *
            m.fs.charge.thermal_oil_disjunct.tank_diameter**2 / 4
        )
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_oil_tank_foundation_cost)

    # Expression to compute the total cost for the thermal oil tank
    m.fs.charge.thermal_oil_disjunct.costing.total_tank_cost = pyo.Expression(
        expr=m.fs.charge.thermal_oil_disjunct.costing.tank_material_cost +
        m.fs.charge.thermal_oil_disjunct.costing.tank_foundation_cost +
        m.fs.charge.thermal_oil_disjunct.costing.tank_insulation_cost
    )

    ###########################################################################
    # Add capital cost: 5. Calculate total capital cost for charge system
    ###########################################################################

    # ---------- Solar salt ----------

    # Add capital cost variable at flowsheet level to handle the
    # storage material capital cost depending on the selected storage
    # material
    m.fs.charge.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e10),
        doc="Annualized capital cost in $ per year")
    m.fs.charge.solar_salt_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Annualized capital cost for Solar salt in $ per year")

    # Calculate annualize capital cost for Solar salt storage system
    def solar_cap_cost_rule(b):
        return m.fs.charge.solar_salt_disjunct.capital_cost == (
            m.fs.charge.solar_salt_disjunct.salt_purchase_cost +
            m.fs.charge.solar_salt_disjunct.spump_purchase_cost +
            (m.fs.charge.solar_salt_disjunct.hxc.costing.purchase_cost +
             m.fs.charge.hx_pump.costing.purchase_cost +
             m.fs.charge.solar_salt_disjunct.no_of_tanks *
             m.fs.charge.solar_salt_disjunct.costing.total_tank_cost) /
            m.fs.charge.num_of_years
        )
    m.fs.charge.solar_salt_disjunct.cap_cost_eq = pyo.Constraint(
        rule=solar_cap_cost_rule)

    # Add constraint to link the global capital cost variable to
    # Solar salt capital cost from disjunction 1
    m.fs.charge.solar_salt_disjunct.fs_cap_cost_eq = pyo.Constraint(
        expr=(
            m.fs.charge.capital_cost ==
            m.fs.charge.solar_salt_disjunct.capital_cost
        )
    )

    # Initialize Solar salt capital cost
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.capital_cost,
        m.fs.charge.solar_salt_disjunct.cap_cost_eq)

    # ---------- Hitec salt ----------
    m.fs.charge.hitec_salt_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Annualized capital cost for Solar salt")

    # Calculate annualized capital cost for Hitec salt storage system
    def hitec_cap_cost_rule(b):
        return m.fs.charge.hitec_salt_disjunct.capital_cost == (
            m.fs.charge.hitec_salt_disjunct.salt_purchase_cost +
            m.fs.charge.hitec_salt_disjunct.spump_purchase_cost +
            (m.fs.charge.hitec_salt_disjunct.hxc.costing.purchase_cost +
             m.fs.charge.hx_pump.costing.purchase_cost +
             m.fs.charge.hitec_salt_disjunct.no_of_tanks *
             m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost)
            / m.fs.charge.num_of_years
        )
    m.fs.charge.hitec_salt_disjunct.cap_cost_eq = pyo.Constraint(
        rule=hitec_cap_cost_rule)

    # Add constraint to link the global capital cost variable to Hitec
    # salt capital cost from disjunction 1
    m.fs.charge.hitec_salt_disjunct.fs_cap_cost_eq = pyo.Constraint(
        expr=(
            m.fs.charge.capital_cost ==
            m.fs.charge.hitec_salt_disjunct.capital_cost
        )
    )

    # Initialize Hitec salt capital cost
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.capital_cost,
        m.fs.charge.hitec_salt_disjunct.cap_cost_eq)

    # ---------- Thermal oil ----------
    m.fs.charge.thermal_oil_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e10),
        doc="Annualized capital cost for thermal oil")

    # Calculate annualized capital cost for the thermal oil storage
    # system
    def oil_cap_cost_rule(b):
        return m.fs.charge.thermal_oil_disjunct.capital_cost == (
            m.fs.charge.thermal_oil_disjunct.salt_purchase_cost +
            m.fs.charge.thermal_oil_disjunct.spump_purchase_cost +
            (m.fs.charge.thermal_oil_disjunct.hxc.costing.purchase_cost +
             m.fs.charge.hx_pump.costing.purchase_cost +
             m.fs.charge.thermal_oil_disjunct.no_of_tanks *
             m.fs.charge.thermal_oil_disjunct.costing.total_tank_cost)
            / m.fs.charge.num_of_years
        )
    m.fs.charge.thermal_oil_disjunct.cap_cost_eq = pyo.Constraint(
        rule=oil_cap_cost_rule)

    # Add constraint to link the global capital cost variable to
    # thermal oil capital cost from disjunction 1
    m.fs.charge.thermal_oil_disjunct.fs_cap_cost_eq = pyo.Constraint(
        expr=(
            m.fs.charge.capital_cost ==
            m.fs.charge.thermal_oil_disjunct.capital_cost
        )
    )

    # Initialize thermal oil capital cost
    calculate_variable_from_constraint(
        m.fs.charge.thermal_oil_disjunct.capital_cost,
        m.fs.charge.thermal_oil_disjunct.cap_cost_eq)

    ###########################################################################
    #  Add operating cost
    ###########################################################################
    m.fs.charge.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.charge.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.charge.operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Operating cost in $ per year")

    def op_cost_rule(b):
        return m.fs.charge.operating_cost == (
            m.fs.charge.operating_hours *
            m.fs.charge.coal_price *
            (m.fs.coal_heat_duty * 1e6) -
            (m.fs.charge.cooling_price *
             m.fs.charge.operating_hours *
             m.fs.charge.cooler.heat_duty[0])
        )
    m.fs.charge.op_cost_eq = pyo.Constraint(rule=op_cost_rule)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.charge.operating_cost,
        m.fs.charge.op_cost_eq)

    ###########################################################################
    #  Add capital and operating cost for full plant
    ###########################################################################

    # Add variables and functions to calculate the plant capital cost
    # and variable and fixed operating costs
    m.fs.charge.plant_capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Annualized capital cost for the plant in $ per year")
    m.fs.charge.plant_fixed_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $ per year")
    m.fs.charge.plant_variable_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $ per year")

    def plant_cap_cost_rule(b):
        return m.fs.charge.plant_capital_cost == (
            (2688973 * m.fs.plant_power_out[0] +
             618968072) /
            m.fs.charge.num_of_years
        ) * (m.CE_index / 575.4)
    m.fs.charge.plant_cap_cost_eq = pyo.Constraint(rule=plant_cap_cost_rule)

    # Initialize capital cost of power plant
    calculate_variable_from_constraint(
        m.fs.charge.plant_capital_cost,
        m.fs.charge.plant_cap_cost_eq)

    def op_fixed_plant_cost_rule(b):
        return m.fs.charge.plant_fixed_operating_cost == (
            (16657.5 * m.fs.plant_power_out[0] +
             6109833.3) /
            m.fs.charge.num_of_years
        ) * (m.CE_index / 575.4)
    m.fs.charge.op_fixed_plant_cost_eq = pyo.Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return m.fs.charge.plant_variable_operating_cost == (
            31754.7 * m.fs.plant_power_out[0]
        ) * (m.CE_index / 575.4)
    m.fs.charge.op_variable_plant_cost_eq = pyo.Constraint(
        rule=op_variable_plant_cost_rule)

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.charge.plant_fixed_operating_cost,
        m.fs.charge.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.charge.plant_variable_operating_cost,
        m.fs.charge.op_variable_plant_cost_eq)

    # Assert 0 degrees of freedom to solve a square problem
    assert degrees_of_freedom(m) == 0

    # Solve cost initialization
    cost_results = solver.solve(m, options=optarg)
    print()
    print("Cost initialization solver termination = ",
          cost_results.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print()
    print()


def calculate_bounds(m):
    """Function to calculate bounds based on the equations in the
    properties package

    """

    # Add a delta temperature
    m.fs.delta_temperature = 5          # Units in K

    # Calculate bounds for Solar salt, Hitec salt, and thermal oil
    # from properties expressions
    m.fs.charge.solar_salt_temperature_max = 853.15 + m.fs.delta_temperature
    m.fs.charge.solar_salt_temperature_min = 513.15 - m.fs.delta_temperature
    m.fs.charge.hitec_salt_temperature_max = 788.15 + m.fs.delta_temperature
    m.fs.charge.hitec_salt_temperature_min = 435.15 - m.fs.delta_temperature
    m.fs.charge.thermal_oil_temperature_max = 616 + m.fs.delta_temperature
    m.fs.charge.thermal_oil_temperature_min = 298.15 - m.fs.delta_temperature

    # Note: min/max interchanged because at max temperature we obtain
    # the mininimum value
    m.fs.charge.solar_salt_enthalpy_mass_max = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.charge.solar_salt_temperature_max - 273.15)) +
        (m.fs.solar_salt_properties.cp_param_2.value * 0.5 *
         (m.fs.charge.solar_salt_temperature_max - 273.15)**2)
    )
    m.fs.charge.solar_salt_enthalpy_mass_min = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.charge.solar_salt_temperature_min - 273.15)) +
        (m.fs.solar_salt_properties.cp_param_2.value * 0.5 *
         (m.fs.charge.solar_salt_temperature_min - 273.15)**2)
    )

    m.fs.charge.hitec_salt_enthalpy_mass_max = (
        (m.fs.hitec_salt_properties.cp_param_1.value *
         (m.fs.charge.hitec_salt_temperature_max)) +
        (m.fs.hitec_salt_properties.cp_param_2.value *
         (m.fs.charge.hitec_salt_temperature_max)**2) +
        (m.fs.hitec_salt_properties.cp_param_3.value *
         (m.fs.charge.hitec_salt_temperature_max)**3)
    )
    m.fs.charge.hitec_salt_enthalpy_mass_min = (
        (m.fs.hitec_salt_properties.cp_param_1.value *
         (m.fs.charge.hitec_salt_temperature_min)) +
        (m.fs.hitec_salt_properties.cp_param_2.value *
         (m.fs.charge.hitec_salt_temperature_min)**2) +
        (m.fs.hitec_salt_properties.cp_param_3.value *
         (m.fs.charge.hitec_salt_temperature_min)**3)
    )

    m.fs.charge.thermal_oil_enthalpy_mass_max = (
        1e3 * (0.003313 *
               (m.fs.charge.thermal_oil_temperature_max - 273.15)**2/2 +
               0.0000008970785 *
               (m.fs.charge.thermal_oil_temperature_max - 273.15)**3/3 +
               1.496005 *
               (m.fs.charge.thermal_oil_temperature_max - 273.15))
    )
    m.fs.charge.thermal_oil_enthalpy_mass_min = (
        1e3 * (0.003313 *
               (m.fs.charge.thermal_oil_temperature_min - 273.15)**2/2 +
               0.0000008970785 *
               (m.fs.charge.thermal_oil_temperature_min - 273.15)**3/3 +
               1.496005 *
               (m.fs.charge.thermal_oil_temperature_min - 273.15))
    )

    m.fs.charge.salt_enthalpy_mass_max = max(m.fs.charge.solar_salt_enthalpy_mass_max,
                                             m.fs.charge.hitec_salt_enthalpy_mass_max)
    m.fs.charge.salt_enthalpy_mass_min = min(m.fs.charge.solar_salt_enthalpy_mass_min,
                                             m.fs.charge.hitec_salt_enthalpy_mass_min)


def add_bounds(m, power_max=None):
    """Add bounds to all units in charge model

    """

    # Add calculated bounds
    calculate_bounds(m)

    # Add maximum and minimum values
    m.factor = 2                          # Scaling factor
    m.flow_max = m.main_flow * 1.2        # Units in mol/s
    m.storage_flow_max = 0.2 * m.flow_max # Units in mol/s
    m.salt_flow_max = 1000                # Units in kg/s
    m.fs.heat_duty_max = 200e6            # Units in MW
    m.power_max = power_max               # Units in MW

    # Add bounds to Solar and Hitec salt charge heat exchangers
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        salt_hxc.inlet_1.flow_mol.setlb(0)
        salt_hxc.inlet_1.flow_mol.setub(m.storage_flow_max)
        salt_hxc.inlet_2.flow_mass.setlb(0)
        salt_hxc.inlet_2.flow_mass.setub(m.salt_flow_max)
        salt_hxc.outlet_1.flow_mol.setlb(0)
        salt_hxc.outlet_1.flow_mol.setub(m.storage_flow_max)
        salt_hxc.outlet_2.flow_mass.setlb(0)
        salt_hxc.outlet_2.flow_mass.setub(m.salt_flow_max)
        salt_hxc.inlet_2.pressure.setlb(101320)
        salt_hxc.inlet_2.pressure.setub(101330)
        salt_hxc.outlet_2.pressure.setlb(101320)
        salt_hxc.outlet_2.pressure.setub(101330)
        salt_hxc.heat_duty.setlb(0)
        salt_hxc.heat_duty.setub(m.fs.heat_duty_max)
        salt_hxc.shell.heat.setlb(-m.fs.heat_duty_max)
        salt_hxc.shell.heat.setub(0)
        salt_hxc.tube.heat.setlb(0)
        salt_hxc.tube.heat.setub(m.fs.heat_duty_max)
        salt_hxc.tube.properties_in[:].enthalpy_mass.setlb(
            m.fs.charge.salt_enthalpy_mass_min / m.factor)
        salt_hxc.tube.properties_in[:].enthalpy_mass.setub(
            m.fs.charge.salt_enthalpy_mass_max * m.factor)
        salt_hxc.tube.properties_out[:].enthalpy_mass.setlb(
            m.fs.charge.salt_enthalpy_mass_min / m.factor)
        salt_hxc.tube.properties_out[:].enthalpy_mass.setub(
            m.fs.charge.salt_enthalpy_mass_max * m.factor)
        salt_hxc.overall_heat_transfer_coefficient.setlb(0)
        salt_hxc.overall_heat_transfer_coefficient.setub(10000)
        salt_hxc.area.setlb(0)
        salt_hxc.area.setub(5000)
        salt_hxc.costing.pressure_factor.setlb(0)
        salt_hxc.costing.pressure_factor.setub(1e5)
        salt_hxc.costing.purchase_cost.setlb(0)
        salt_hxc.costing.purchase_cost.setub(1e7)
        salt_hxc.costing.base_cost_per_unit.setlb(0)
        salt_hxc.costing.base_cost_per_unit.setub(1e6)
        salt_hxc.costing.material_factor.setlb(0)
        salt_hxc.costing.material_factor.setub(10)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_in.setub(87)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_out.setub(84)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_in.setlb(10)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_out.setlb(9)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_in.setub(81)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_out.setub(90)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_in.setlb(10)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_out.setlb(10)

    # Add bounds to thermal oil charge heat exchanger
    for oil_hxc in [m.fs.charge.thermal_oil_disjunct.hxc]:
        oil_hxc.inlet_1.flow_mol.setlb(0)
        oil_hxc.inlet_1.flow_mol.setub(m.storage_flow_max)
        oil_hxc.outlet_1.flow_mol.setlb(0)
        oil_hxc.outlet_1.flow_mol.setub(m.storage_flow_max)
        oil_hxc.inlet_2.flow_mass.setlb(0)
        oil_hxc.inlet_2.flow_mass.setub(m.salt_flow_max)
        oil_hxc.outlet_2.flow_mass.setlb(0)
        oil_hxc.outlet_2.flow_mass.setub(m.salt_flow_max)
        oil_hxc.inlet_2.pressure.setlb(101320)
        oil_hxc.inlet_2.pressure.setub(101330)
        oil_hxc.outlet_2.pressure.setlb(101320)
        oil_hxc.outlet_2.pressure.setub(101330)
        oil_hxc.heat_duty.setlb(0)
        oil_hxc.heat_duty.setub(m.fs.heat_duty_max)
        oil_hxc.shell.heat.setlb(-m.fs.heat_duty_max)
        oil_hxc.shell.heat.setub(0)
        oil_hxc.tube.heat.setlb(0)
        oil_hxc.tube.heat.setub(m.fs.heat_duty_max)
        oil_hxc.overall_heat_transfer_coefficient.setlb(0)
        oil_hxc.overall_heat_transfer_coefficient.setub(10000)
        oil_hxc.area.setlb(0)
        oil_hxc.area.setub(8000)
        oil_hxc.tube.properties_in[:].enthalpy_mass.setlb(
            m.fs.charge.thermal_oil_enthalpy_mass_min / m.factor)
        oil_hxc.tube.properties_in[:].enthalpy_mass.setub(
            m.fs.charge.thermal_oil_enthalpy_mass_max * m.factor)
        oil_hxc.tube.properties_out[:].enthalpy_mass.setlb(
            m.fs.charge.thermal_oil_enthalpy_mass_min / m.factor)
        oil_hxc.tube.properties_out[:].enthalpy_mass.setub(
            m.fs.charge.thermal_oil_enthalpy_mass_max * m.factor)
        oil_hxc.costing.pressure_factor.setlb(0)
        oil_hxc.costing.pressure_factor.setub(1e5)
        oil_hxc.costing.purchase_cost.setlb(0)
        oil_hxc.costing.purchase_cost.setub(1e7)
        oil_hxc.costing.base_cost_per_unit.setlb(0)
        oil_hxc.costing.base_cost_per_unit.setub(1e6)
        oil_hxc.costing.material_factor.setlb(0)
        oil_hxc.costing.material_factor.setub(10)
        oil_hxc.delta_temperature_in.setlb(10)
        oil_hxc.delta_temperature_in.setub(554)
        oil_hxc.delta_temperature_out.setlb(9)
        oil_hxc.delta_temperature_out.setub(222)

    # Add bounds to charge storage pump and cooler
    for unit_k in [m.fs.charge.connector,
                   m.fs.charge.hx_pump,
                   m.fs.charge.cooler]:
        unit_k.inlet.flow_mol.setlb(0)
        unit_k.inlet.flow_mol.setub(m.storage_flow_max)
        unit_k.outlet.flow_mol.setlb(0)
        unit_k.outlet.flow_mol.setub(m.storage_flow_max)
    m.fs.charge.cooler.heat_duty.setub(0)

    # Add bounds to cost-related terms
    m.fs.charge.hx_pump.costing.purchase_cost.setlb(0)
    m.fs.charge.hx_pump.costing.purchase_cost.setub(1e7)

    # Add bounds needed in units declared in steam source disjunction
    for split in [m.fs.charge.vhp_source_disjunct.ess_vhp_split,
                  m.fs.charge.hp_source_disjunct.ess_hp_split]:
        split.to_hxc.flow_mol[:].setlb(0)
        split.to_hxc.flow_mol[:].setub(m.storage_flow_max)
        split.to_turbine.flow_mol[:].setlb(0)
        split.to_turbine.flow_mol[:].setub(m.flow_max)
        split.split_fraction[0.0, "to_hxc"].setlb(0)
        split.split_fraction[0.0, "to_hxc"].setub(1)
        split.split_fraction[0.0, "to_turbine"].setlb(0)
        split.split_fraction[0.0, "to_turbine"].setub(1)
        split.inlet.flow_mol[:].setlb(0)
        split.inlet.flow_mol[:].setub(m.flow_max)

    for mix in [m.fs.charge.recycle_mixer]:
        mix.from_bfw_out.flow_mol.setlb(0)
        mix.from_bfw_out.flow_mol.setub(m.flow_max)
        mix.from_hx_pump.flow_mol.setlb(0)
        mix.from_hx_pump.flow_mol.setub(m.storage_flow_max)
        mix.outlet.flow_mol.setlb(0)
        mix.outlet.flow_mol.setub(m.flow_max)

    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)
    m.fs.charge.hx_pump.control_volume.work[0].setlb(0)
    m.fs.charge.hx_pump.control_volume.work[0].setub(1e10)

    m.fs.plant_power_out[0].setlb(300)
    m.fs.plant_power_out[0].setub(m.power_max)

    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)
        unit_k.inlet.flow_mol[:].setub(m.flow_max)
        unit_k.outlet.flow_mol[:].setlb(0)
        unit_k.outlet.flow_mol[:].setub(m.flow_max)

    # Adding bounds on turbine splitters flow
    for k in m.set_turbine_splitter:
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setub(m.flow_max)

    return m


def main(m_usc):

    # Add boiler and cycle efficiency to the model
    add_efficiency = True

    # Add maximum power produced by power plant in MW
    power_max = 436

    # Create flowsheet, add properties, unit models, and arcs
    m = create_charge_model(m_usc, add_efficiency=add_efficiency, power_max=power_max)

    # Give all the required inputs to the model
    set_model_input(m)

    # Add scaling factor
    set_scaling_factors(m)

    # Initialize the model with a sequential initialization and custom
    # routines
    initialize(m)

    # Add cost correlations
    build_costing(m, solver=solver)

    # Add bounds
    add_bounds(m, power_max=power_max)

    # Add disjunctions
    add_disjunction(m)

    return m, solver

def print_model(nlp_model, _):
    """Print the disjunction selected during the solution of the NLP
    subproblem

    """

    nlp = nlp_model.fs.charge
    print('    ___________________________________________')
    print('     Disjunction 1:')
    if nlp.solar_salt_disjunct.indicator_var.value == 1:
        print('      Solar salt is selected')
    elif nlp.hitec_salt_disjunct.indicator_var.value == 1:
        print('      Hitec salt is selected')
    elif nlp.thermal_oil_disjunct.indicator_var.value == 1:
        print('      Thermal oil is selected')
    else:
        print('      Error: There are no more alternatives')
    print()
    print('     Disjunction 2:')
    if nlp.vhp_source_disjunct.indicator_var.value == 1:
        print('      Very high-pressure source is selected')
    elif nlp.hp_source_disjunct.indicator_var.value == 1:
        print('      High-pressure source is selected')
    else:
        print('      Error: There are no more alternatives')

    print('    ___________________________________________')
    print()


def run_gdp(m):
    """Declare solver GDPopt and its options
    """

    # Add options to GDPopt
    opt = SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'LOA'
    opt.CONFIG.mip_solver = 'cbc'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.tee = True
    opt.CONFIG.init_strategy = "no_init"
    opt.CONFIG.call_after_subproblem_solve = print_model
    opt.CONFIG.nlp_solver_args.tee = True
    opt.CONFIG.subproblem_presolve = False
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None

    # Solve model
    results = opt.solve(m)

    return results


def print_results(m, results):

    print()
    print('====================================================')
    print('Results ')
    print()
    print('Objective function (M$/year): {:.2f}'.format(
        (pyo.value(m.obj) / scaling_obj) * 1e-6))
    print('Charge capital cost (M$/year): {:.6f}'.format(
        value(m.fs.charge.capital_cost) * 1e-6))
    print('Plant Power (MW): {:.2f}'.format(
        pyo.value(m.fs.plant_power_out[0])))
    print('Boiler Efficiency (%): {:.2f}'.format(
        pyo.value(m.fs.boiler_efficiency) * 100))
    print()
    print('**Discrete design decisions (Disjunctions)')
    for d in m.component_data_objects(ctype=Disjunct,
                                      active=True,
                                      sort=True, descend_into=True):
        if abs(d.indicator_var.value - 1) < 1e-6:
            print(d.name, 'is selected!')
    if m.fs.charge.solar_salt_disjunct.indicator_var.value == 1:
        print('Solar salt heat exchanger area (m2): {:.2f}'.format(
            pyo.value(m.fs.charge.solar_salt_disjunct.hxc.area)))
        print('Solar salt heat exchanger heat duty (MW): {:.2f}'.format(
            pyo.value(m.fs.charge.solar_salt_disjunct.hxc.heat_duty[0]) / 1e6))
    elif m.fs.charge.hitec_salt_disjunct.indicator_var.value == 1:
        print('Hitec salt heat exchanger area (m2): {:.2f}'.format(
            pyo.value(m.fs.charge.hitec_salt_disjunct.hxc.area)))
        print('Hitec salt heat exchanger heat duty (MW): {:.2f}'.format(
            pyo.value(m.fs.charge.hitec_salt_disjunct.hxc.heat_duty[0]) / 1e6))
    elif m.fs.charge.thermal_oil_disjunct.indicator_var.value == 1:
        print('Thermal oil heat exchanger area (m2): {:.2f}'.format(
            pyo.value(m.fs.charge.thermal_oil_disjunct.hxc.area)))
        print('Thermal oil heat exchanger heat duty (MW): {:.2f}'.format(
            pyo.value(m.fs.charge.thermal_oil_disjunct.hxc.heat_duty[0]) * 1e-6))
    else:
        print('No other alternatives!')
    print('====================================================')
    print()
    print('Solver details')
    print(results)


def model_analysis(m, heat_duty=None):
    """Solve the conceptual design optimization problem

    """

    # Fix variables in the flowsheet
    m.fs.plant_power_out.fix(400)
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.charge.solar_salt_disjunct.hxc.heat_duty.fix(heat_duty * 1e6)
    m.fs.charge.hitec_salt_disjunct.hxc.heat_duty.fix(heat_duty * 1e6)
    m.fs.charge.thermal_oil_disjunct.hxc.heat_duty.fix(heat_duty * 1e6)

    # Unfix variables that were fixed during initialization
    m.fs.boiler.inlet.flow_mol.unfix()
    m.fs.charge.vhp_source_disjunct.ess_vhp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.charge.hp_source_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.charge.hx_pump.outlet.pressure[0].unfix()
    m.fs.turbine[1].inlet.unfix()
    m.fs.turbine[3].inlet.unfix()

    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc,
                     m.fs.charge.thermal_oil_disjunct.hxc]:
        salt_hxc.inlet_1.unfix()
        salt_hxc.inlet_2.flow_mass.unfix()
        salt_hxc.area.unfix()

    for unit in [m.fs.charge.connector,
                 m.fs.charge.cooler]:
        unit.inlet.unfix()
    m.fs.charge.cooler.outlet.enth_mol[0].unfix()

    # Deactivate production constraint from ultra-supercritical plant
    # base model and include a new constraint that includes the
    # storage pump power
    m.fs.production_cons.deactivate()
    @m.fs.charge.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return (
            (-1 * sum(m.fs.turbine[p].work_mechanical[t]
                      for p in m.set_turbine)
             - m.fs.charge.hx_pump.control_volume.work[0]
            ) ==
            m.fs.plant_power_out[t] * 1e6 * (pyunits.W/pyunits.MW)
        )

    # Add total cost as the objective function
    m.obj = Objective(
        expr=(
            m.fs.charge.capital_cost
            + m.fs.charge.operating_cost
            + m.fs.charge.plant_capital_cost
            + m.fs.charge.plant_fixed_operating_cost
            + m.fs.charge.plant_variable_operating_cost
        ) * scaling_obj
    )



if __name__ == "__main__":

    optarg = {"max_iter": 300}
    solver = get_solver('ipopt', optarg)

    heat_duty_data = 150

    # Build ultra-supercritical plant base model
    m_usc = usc.build_plant_model()

    # Initialize ultra-supercritical plant base model
    usc.initialize(m_usc)

    # Build charge model
    m, solver = main(m_usc)

    # Solve design optimization problem
    model_analysis(m, heat_duty=heat_duty_data)

    # Solve model using GDPopt
    print()
    print('**********Start solution of charge GDP model using GDPopt')
    print('>>DOFs before solution of charge GDP model: ', degrees_of_freedom(m))
    print()
    results = run_gdp(m)

    # Print results
    print_results(m, results)
