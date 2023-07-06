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

"""This is a GDP model for the conceptual design of an ultra
supercritical coal-fired power plant based on a flowsheet presented in
1999 USDOE Report #DOE/FE-0400

This model uses some of the simpler unit models from the power
generation unit model library.

Some of the parameters in the model such as feed water heater areas,
overall heat transfer coefficient, turbine efficiencies at multiple
stages have all been estimated for a total power out of 437 MW.

Additional main assumptions are as follows:
(1) The flowsheet and main steam conditions, i. e. pressure &
    temperature are adopted from the aforementioned DOE report
(2) Heater unit models are used to model main steam boiler, reheater,
    and condenser.  
(3) Multi-stage turbines are modeled as multiple
    lumped single stage turbines

"""

# Notes by esrawli:
# In this version of the model, the following changes were made:
# 1. Use thermal oil updated properties that were re-written to have expressions instead of variables
# 2. The production constraint (to calculate plant_power_out) now includes the missing hx pump work
# 3. The hx pump pressure out constraint is an inequality constraint instead of an equality constraint
# 4. The VHP and HP splitters for the steam source disjunction are now include inside each disjunct
# 5. New disjunction to include or not a cooler
# 6. Corrected constraints:
#    - op_cost_rule, without the -q_baseline
#    - plant_cap_cost_rule, op_fixed_cap_cost_rule, op_variable_cap_cost_rule
#      using plant_power_out instead of plant_heat_duty and multiply by (CE_index/575.4)
# 7. Number of years was changed from 5 to 30
# 8. Add boiler and cycle efficiency
# 9. Add revenue expression and random lmp signal value
# 10. Objective function is a maximization of profit

__author__ = "Naresh Susarla and Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
import os
from IPython import embed
import csv

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective,
                           TransformationFactory, SolverFactory,
                           Expression, value, log, exp, Var, maximize)
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
from pyomo.network.plugins import expand_arcs
from pyomo.contrib.fbbt.fbbt import  _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core import UnitModelCostingBlock
from idaes.models.unit_models import (HeatExchanger,
                                      MomentumMixingType,
                                      Heater)
from idaes.models.unit_models import (Mixer,
                                      PressureChanger)
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmIsentropicCompressor,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)
from idaes.models.unit_models.separator import (Separator,
                                                SplittingType)
from idaes.models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback, HeatExchangerFlowPattern)
from idaes.models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
from idaes.models.costing.SSLW import (
    SSLWCosting,
    SSLWCostingData,
    PumpType,
    PumpMaterial,
    PumpMotorType,
)
from idaes.core.util.exceptions import ConfigurationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import ultra supercritical power plant model
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

# Import properties package for storage materials
from dispatches.properties import (solarsalt_properties,
                                   hitecsalt_properties,
                                   thermaloil_properties)

logging.basicConfig(level=logging.INFO)
logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)

scaling_obj = 1
save_csv = True

print("Using scaling_obj={}".format(scaling_obj))

def create_charge_model(m, method=None, max_power=None):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    m.fs.charge = Block()

    # Add model data
    _add_data(m)

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()
    m.fs.hitec_salt_properties = hitecsalt_properties.HitecsaltParameterBlock()
    m.fs.therminol66_properties = thermaloil_properties.ThermalOilParameterBlock()

    ###########################################################################
    #  Add a dummy heat exchanger                                  #
    ###########################################################################
    # A connector model is defined as a dummy heat exchanger with Q=0
    # and a deltaP=0
    m.fs.charge.connector = Heater(
        property_package=m.fs.prop_water,
        has_pressure_change=False,
    )

    ###########################################################################
    #  Add cooler connector and HX pump                                       #
    ###########################################################################
    # Declare a cooler connector as a dummy heat exchanger with Q=0
    # and a deltaP=0
    m.fs.charge.cooler_connector = Heater(
        property_package=m.fs.prop_water,
        has_pressure_change=False,
    )

    # A pump, if needed, is used to increase the pressure of the water
    # to allow mixing it at a desired location within the plant
    m.fs.charge.hx_pump = PressureChanger(
        property_package=m.fs.prop_water,
        material_balance_type=MaterialBalanceType.componentTotal,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    ###########################################################################
    #  Add variables to global model
    ###########################################################################
    m.fs.charge.cooler_heat_duty = pyo.Var(m.fs.time,
                                       doc="Cooler heat duty in W",
                                       bounds=(-1e10, 0),
                                       initialize=0)
    m.fs.charge.cooler_capital_cost = pyo.Var(bounds=(0, 1e8),
                                          doc="Annualized cooler capital cost in $/y",
                                          initialize=0)

    ###########################################################################
    #  Declare disjuncts
    ###########################################################################
    # Disjunction 1 for the storage fluid selection consists of 2 disjuncts:
    #   1. solar_salt_disjunct ======> solar salt used as the storage material
    #   2. hitec_salt_disjunct ======> hitec salt used as the storage material
    #   3. thermal_oil_disjunct =====> thermal oil used as the storage material
    # Disjunction 2 for the steam source selection consists of 2 disjuncts:
    #   1. vhp_source_disjunct ===> high pressure steam for heat source
    #   2. hp_source_disjunct ===> intermediate pressure steam for heat source
    # Disjunction 3 for the selection of cooler
    #   1. cooler_disjunct ===> include a cooler in storage system
    #   2. no_cooler_disjunct ===> no cooler in storage system
    # Disjunction 4 for the selection of sink
    #   1. recycle_mixer1_sink_disjunct ===> returned condensed steam to recycle mixer 1
    #   2. recycle_mixer2_sink_disjunct ===> returned condensed steam to recycle mixer 2
    #   3. recycle_mixer3_sink_disjunct ===> returned condensed steam to recycle mixer 3
    #   4. recycle_mixer4_sink_disjunct ===> returned condensed steam to recycle mixer 4
    #   5. recycle_mixer5_sink_disjunct ===> returned condensed steam to recycle mixer 5


    m.fs.charge.solar_salt_disjunct = Disjunct(
        rule=solar_salt_disjunct_equations)
    m.fs.charge.hitec_salt_disjunct = Disjunct(
        rule=hitec_salt_disjunct_equations)
    m.fs.charge.thermal_oil_disjunct = Disjunct(
        rule=thermal_oil_disjunct_equations)

    m.fs.charge.vhp_source_disjunct = Disjunct(
        rule=vhp_source_disjunct_equations)
    m.fs.charge.hp_source_disjunct = Disjunct(
        rule=hp_source_disjunct_equations)

    #  Disjunction 3
    m.fs.charge.cooler_disjunct = Disjunct(
        rule=cooler_disjunct_equations)
    m.fs.charge.no_cooler_disjunct = Disjunct(
        rule=no_cooler_disjunct_equations)

    # Disjunction 4
    m.fs.charge.recycle_mixer1_sink_disjunct = Disjunct(
        rule=recycle_mixer1_sink_disjunct_equations)
    m.fs.charge.recycle_mixer2_sink_disjunct = Disjunct(
        rule=recycle_mixer2_sink_disjunct_equations)
    m.fs.charge.recycle_mixer3_sink_disjunct = Disjunct(
        rule=recycle_mixer3_sink_disjunct_equations)
    m.fs.charge.recycle_mixer4_sink_disjunct = Disjunct(
        rule=mixer1_sink_disjunct_equations)
    m.fs.charge.recycle_mixer5_sink_disjunct = Disjunct(
        rule=mixer2_sink_disjunct_equations)

    ###########################################################################
    # Add constraints and create the stream Arcs and return the model
    ###########################################################################
    _make_constraints(m, method=method, max_power=max_power)

    _disconnect_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)
    return m


def _add_data(m):
    """Add data to the model
    """

    # Add Chemical engineering cost index for 2019
    m.CE_index = 607.5  # Chemical engineering cost index for 2019

    # Add number of operating hours
    m.fs.charge.hours_per_day = pyo.Param(
        initialize=6,
        doc='Estimated number of hours of charging per day'
    )

    # Add number of years over which the capital cost is annualized
    m.fs.charge.num_of_years = pyo.Param(
        initialize=30,
        doc='Number of years for capital cost annualization')

    # Add data to compute overall heat transfer coefficient for the
    # Solar salt, Hitec salt, and Thermal oil storage heat exchangers
    # using the Sieder-Tate Correlation. Parameters for tube diameter
    # and thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985
    m.fs.charge.data_hxc = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    m.fs.charge.hxc_tube_thickness = pyo.Param(
        initialize=m.fs.charge.data_hxc['tube_thickness'],
        doc='Tube thickness in m')
    m.fs.charge.hxc_tube_inner_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['tube_inner_dia'],
        doc='Tube inner diameter in m')
    m.fs.charge.hxc_tube_outer_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['tube_outer_dia'],
        doc='Tube outer diameter in m')
    m.fs.charge.hxc_k_steel = pyo.Param(
        initialize=m.fs.charge.data_hxc['k_steel'],
        doc='Thermal conductivity of steel in W/m.K')
    m.fs.charge.hxc_n_tubes = pyo.Param(
        initialize=m.fs.charge.data_hxc['number_tubes'],
        doc='Number of tubes ')
    m.fs.charge.hxc_shell_inner_dia = pyo.Param(
        initialize=m.fs.charge.data_hxc['shell_inner_dia'],
        doc='Shell inner diameter in m')

    m.fs.charge.hxc_tube_cs_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.charge.hxc_tube_inner_dia ** 2),
        doc="Tube inside cross sectional area in m2")
    m.fs.charge.hxc_tube_out_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.charge.hxc_tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness in m2")
    m.fs.charge.hxc_shell_eff_area = pyo.Expression(
        expr=(
            (pi / 4) *
            (m.fs.charge.hxc_shell_inner_dia ** 2) -
            m.fs.charge.hxc_n_tubes *
            m.fs.charge.hxc_tube_out_area),
        doc="Effective shell cross sectional area in m2")

    m.fs.charge.hxc_tube_dia_ratio = (m.fs.charge.hxc_tube_outer_dia /
                                      m.fs.charge.hxc_tube_inner_dia)
    m.fs.charge.hxc_log_tube_dia_ratio = log(m.fs.charge.hxc_tube_dia_ratio)


    # Add fuel and storage material cost data
    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'solar_salt_price': 0.49,
        'hitec_salt_price': 0.93,
        'thermal_oil_price': 6.72,  # $/kg
        'storage_tank_material': 3.5,
        'storage_tank_insulation': 235,
        'storage_tank_foundation': 1210
    }

    m.fs.charge.coal_price = Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')
    m.fs.charge.cooling_price = Param(
        initialize=m.data_cost['cooling_price'],
        doc='Cost of chilled water for cooler from Sieder et al. $/J')
    m.fs.charge.solar_salt_price = Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')
    m.fs.charge.hitec_salt_price = Param(
        initialize=m.data_cost['hitec_salt_price'],
        doc='Hitec salt price in $/kg')
    m.fs.charge.thermal_oil_price = Param(
        initialize=m.data_cost['thermal_oil_price'],
        doc='Thermal oil price in $/kg')
    m.fs.charge.storage_tank_material_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_material'],
        doc='$/kg of SS316 material')
    m.fs.charge.storage_tank_insulation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_insulation'],
        doc='$/m2')
    m.fs.charge.storage_tank_foundation_cost = pyo.Param(
        initialize=m.data_cost['storage_tank_foundation'],
        doc='$/m2')

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
        doc='Storage tank thickness assumed based on reference'
    )
    m.fs.charge.storage_tank_material_dens_mass = pyo.Param(
        initialize=m.data_storage_tank['material_density'],
        doc='Kg/m3')

    # Add parameters to calculate salt and oil pump costing. Since the
    # pump units are not explicitly modeled, the IDAES cost method is
    # not used for this equipment.  The primary purpose of the salt and oil
    # pump is to move the storage material without changing the pressure.
    # Thus, the pressure head is computed assuming that the salt or oil
    # is moved on an average of 5 m (i.e., 16.41 ft) linear distance.
    m.data_salt_pump = {
        'FT': 1.5,
        'FM': 2.0,
        'head': 3.281*5, # in ft, equivalent to 5 m
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


def _make_constraints(m, method=None, max_power=None):
    """Declare the constraints for the charge model
    """

    # HX pump
    @m.fs.Constraint(m.fs.time,
                     doc="HX pump out pressure equal to BFP out pressure")
    def constraint_hxpump_presout(b, t):
        return b.charge.hx_pump.outlet.pressure[t] >= \
            (m.main_steam_pressure * 1.1231)

    m.fs.max_power = Param(
        initialize=max_power,
        mutable=True,
        doc='Pmax for the power plant [MW]')
    m.fs.boiler_efficiency = Expression(
        expr=0.2143
        * (m.fs.plant_power_out[0] / m.fs.max_power)
        + 0.7357,
        doc="Boiler efficiency in fraction"
    )
    m.fs.coal_heat_duty = pyo.Var(
        initialize=1000,
        bounds=(0, 1e5),
        doc="Coal heat duty supplied to boiler (MW)")

    if method == "with_efficiency":
        def coal_heat_duty_rule(b):
            return b.coal_heat_duty * b.boiler_efficiency == (
                b.plant_heat_duty[0])
        m.fs.coal_heat_duty_eq = Constraint(rule=coal_heat_duty_rule)

    else:
        def coal_heat_duty_rule(b):
            return b.coal_heat_duty == (
                b.plant_heat_duty[0])
        m.fs.coal_heat_duty_eq = Constraint(rule=coal_heat_duty_rule)

    m.fs.cycle_efficiency = Expression(
        expr=(m.fs.plant_power_out[0] / m.fs.coal_heat_duty),
        doc="Cycle efficiency"
    )


def _disconnect_arcs(m):
    """Disconnect arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the storage system. ** Note ** Some arcs are
    # deactivated in model_analysis to avoid changing the
    # initialization and having to re-initialize some units (because
    # of DOFs)
    for arc_s in [m.fs.boiler_to_turb1,
                  m.fs.bfp_to_fwh8,
                  m.fs.rh1_to_turb3,
                  m.fs.fwh8_to_fwh9,
                  m.fs.fwh9_to_boiler, # disconnected during design optimization
                  m.fs.fwh6_to_fwh7,
                  m.fs.booster_to_fwh6]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def add_disjunction(m):
    """Add storage fluid selection and steam source disjunctions to the
    model
    """

    m.fs.cooler_disjunction = Disjunction(
        expr=[m.fs.charge.cooler_disjunct,
              m.fs.charge.no_cooler_disjunct])


    # Add disjunction 1 for the storage fluid selection
    m.fs.salt_disjunction = Disjunction(
        expr=[m.fs.charge.solar_salt_disjunct,
              m.fs.charge.hitec_salt_disjunct,
              m.fs.charge.thermal_oil_disjunct]
    )

    # Add disjunction 2 for the source selection
    m.fs.source_disjunction = Disjunction(
        expr=[m.fs.charge.vhp_source_disjunct,
              m.fs.charge.hp_source_disjunct]
    )

    # Add disjunction 4 for the sink selection
    m.fs.sink_disjunction = Disjunction(
        expr=[m.fs.charge.recycle_mixer3_sink_disjunct,
              m.fs.charge.recycle_mixer4_sink_disjunct,
              m.fs.charge.recycle_mixer5_sink_disjunct,
              m.fs.charge.recycle_mixer2_sink_disjunct,
              m.fs.charge.recycle_mixer1_sink_disjunct])


    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.charge)

    return m


def solar_salt_disjunct_equations(disj):
    """Block of equations for disjunct 1 for the selection of solar salt
    as the storage fluid in charge heat exchanger
    """

    m = disj.model()

    # Add solar salt heat exchanger
    m.fs.charge.solar_salt_disjunct.hxc = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.prop_water},
        tube={"property_package": m.fs.solar_salt_properties},
    )

    # Calculate heat transfer coefficient for solar salt heat
    # exchanger. For that, calculate first the Reynolds, Prandtl, and
    # Nusselt number for the salt and steam side of hitec charge heat
    # exchanger
    solar_hxc = m.fs.charge.solar_salt_disjunct.hxc
    solar_hxc.salt_reynolds_number = Expression(
        expr=(
            (solar_hxc.tube_inlet.flow_mass[0] *
             m.fs.charge.hxc_tube_outer_dia) /
            (m.fs.charge.hxc_shell_eff_area *
             solar_hxc.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number")
    solar_hxc.salt_prandtl_number = Expression(
        expr=(
            solar_hxc.cold_side.properties_in[0].cp_mass["Liq"] *
            solar_hxc.cold_side.properties_in[0].visc_d_phase["Liq"] /
            solar_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number")
    solar_hxc.salt_prandtl_wall = Expression(
        expr=(
            solar_hxc.cold_side.properties_out[0].cp_mass["Liq"] *
            solar_hxc.cold_side.properties_out[0].visc_d_phase["Liq"] /
            solar_hxc.cold_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    solar_hxc.salt_nusselt_number = Expression(
        expr=(
            0.35 *
            (solar_hxc.salt_reynolds_number**0.6) *
            (solar_hxc.salt_prandtl_number**0.4) *
            ((solar_hxc.salt_prandtl_number /
              solar_hxc.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    solar_hxc.steam_reynolds_number = Expression(
        expr=(
            solar_hxc.shell_inlet.flow_mol[0] *
            solar_hxc.hot_side.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia /
            (m.fs.charge.hxc_tube_cs_area *
             m.fs.charge.hxc_n_tubes *
             solar_hxc.hot_side.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")
    solar_hxc.steam_prandtl_number = Expression(
        expr=(
            (solar_hxc.hot_side.properties_in[0].cp_mol /
             solar_hxc.hot_side.properties_in[0].mw) *
            solar_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
            solar_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    solar_hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (solar_hxc.steam_reynolds_number**0.8) *
            (solar_hxc.steam_prandtl_number**(0.33)) *
            ((solar_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
              solar_hxc.hot_side.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of charge heat exchanger
    solar_hxc.h_salt = Expression(
        expr=(
            solar_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"] *
            solar_hxc.salt_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    solar_hxc.h_steam = Expression(
        expr=(
            solar_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"] *
            solar_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    # Calculate overall heat transfer coefficient for Solar salt
    # charge heat exchanger
    @m.fs.charge.solar_salt_disjunct.hxc.Constraint(m.fs.time)
    def constraint_hxc_ohtc(b, t):
        return (
            b.overall_heat_transfer_coefficient[t] *
            (2 *
             m.fs.charge.hxc_k_steel *
             b.h_steam +
             m.fs.charge.hxc_tube_outer_dia *
             m.fs.charge.hxc_log_tube_dia_ratio *
             b.h_salt *
             b.h_steam +
             m.fs.charge.hxc_tube_dia_ratio *
             b.h_salt *
             2 * m.fs.charge.hxc_k_steel)
        ) == (2 * m.fs.charge.hxc_k_steel *
              b.h_salt *
              b.h_steam)

    # Declare arcs within the disjunct
    m.fs.charge.solar_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=m.fs.charge.solar_salt_disjunct.hxc.shell_inlet,
        doc="Connection from connector to solar charge heat exchanger"
    )
    m.fs.charge.solar_salt_disjunct.hxc_to_coolconnector = Arc(
        source=m.fs.charge.solar_salt_disjunct.hxc.shell_outlet,
        destination=m.fs.charge.cooler_connector.inlet,
        doc="Connection from solar charge heat exchanger to cooler connector"
    )


def hitec_salt_disjunct_equations(disj):
    """Block of equations for disjunct 2 for the selection of hitec salt
    as the storage medium in charge heat exchanger

    """

    m = disj.model()

    # Declare hitec salt heat exchanger
    m.fs.charge.hitec_salt_disjunct.hxc = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.prop_water},
        tube={"property_package": m.fs.hitec_salt_properties},
    )

    # Calculate heat transfer coefficient for hitec salt heat
    # exchanger. For that, calculate first the Reynolds, Prandtl, and
    # Nusselt number for the salt and steam side of hitec charge heat
    # exchanger
    hitec_hxc = m.fs.charge.hitec_salt_disjunct.hxc
    hitec_hxc.salt_reynolds_number = Expression(
        expr=(
            hitec_hxc.tube_inlet.flow_mass[0] *
            m.fs.charge.hxc_tube_outer_dia /
            (m.fs.charge.hxc_shell_eff_area *
             hitec_hxc.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    hitec_hxc.salt_prandtl_number = Expression(
        expr=(
            hitec_hxc.cold_side.properties_in[0].cp_mass["Liq"]
            * hitec_hxc.cold_side.properties_in[0].visc_d_phase["Liq"]
            / hitec_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]),
        doc="Salt Prandtl Number")
    hitec_hxc.salt_prandtl_wall = Expression(
        expr=(
            hitec_hxc.cold_side.properties_out[0].cp_mass["Liq"]
            * hitec_hxc.cold_side.properties_out[0].visc_d_phase["Liq"]
            / hitec_hxc.cold_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Wall Prandtl Number"
    )
    hitec_hxc.salt_nusselt_number = Expression(
        expr=(
            1.61 * ((hitec_hxc.salt_reynolds_number *
                     hitec_hxc.salt_prandtl_number *0.009)**0.63) *
            ((hitec_hxc.cold_side.properties_in[0].visc_d_phase["Liq"] /
              hitec_hxc.cold_side.properties_out[0].visc_d_phase["Liq"])**0.25)
        ),
        doc="Salt Nusslet Number from 2014, He et al, Exp Therm Fl Sci, 59, 9"
    )
    hitec_hxc.steam_reynolds_number = Expression(
        expr=(
            hitec_hxc.shell_inlet.flow_mol[0] *
            hitec_hxc.hot_side.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia
            / (m.fs.charge.hxc_tube_cs_area *
               m.fs.charge.hxc_n_tubes *
               hitec_hxc.hot_side.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    hitec_hxc.steam_prandtl_number = Expression(
        expr=(
            (hitec_hxc.hot_side.properties_in[0].cp_mol /
             hitec_hxc.hot_side.properties_in[0].mw) *
            hitec_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
            hitec_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    hitec_hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (hitec_hxc.steam_reynolds_number** 0.8) *
            (hitec_hxc.steam_prandtl_number** 0.33) *
            ((hitec_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
              hitec_hxc.hot_side.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Calculate heat transfer coefficient for salt and steam side of
    # charge heat exchanger
    hitec_hxc.h_salt = Expression(
        expr=(
            hitec_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"] *
            hitec_hxc.salt_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient in W/m.K"
    )
    hitec_hxc.h_steam = Expression(
        expr=(
            hitec_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"] *
            hitec_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient in W/m.K"
    )

    # Calculate overall heat transfer coefficient for Solar salt
    # charge heat exchanger
    @m.fs.charge.hitec_salt_disjunct.hxc.Constraint(
        m.fs.time,
        doc="Hitec salt charge heat exchanger overall heat transfer coefficient")
    def constraint_hxc_ohtc_hitec(b, t):
        return (
            b.overall_heat_transfer_coefficient[t] *
            (2 *
             m.fs.charge.hxc_k_steel *
             b.h_steam
             + m.fs.charge.hxc_tube_outer_dia *
             m.fs.charge.hxc_log_tube_dia_ratio *
             b.h_salt *
             b.h_steam
             + m.fs.charge.hxc_tube_dia_ratio *
             b.h_salt *
             2 * m.fs.charge.hxc_k_steel)
        ) == (2 * m.fs.charge.hxc_k_steel *
              b.h_salt *
              b.h_steam)

    # Declare arcs to connect units within the disjunct
    m.fs.charge.hitec_salt_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=m.fs.charge.hitec_salt_disjunct.hxc.shell_inlet,
        doc="Connect the connector to hitec heat exchanger"
    )
    m.fs.charge.hitec_salt_disjunct.hxc_to_coolconnector = Arc(
        source=m.fs.charge.hitec_salt_disjunct.hxc.shell_outlet,
        destination=m.fs.charge.cooler_connector.inlet,
        doc="Connect hitec charge heat exchanger to cooler connector"
    )


def thermal_oil_disjunct_equations(disj):
    """Block of equations for disjunct 2 for the selection of thermal oil
    as the storage medium in charge heat exchanger

    """

    m = disj.model()

    # Declare thermal oil heat exchanger
    m.fs.charge.thermal_oil_disjunct.hxc = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.prop_water},
        tube={"property_package": m.fs.therminol66_properties},
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
    )

    # Calculate heat transfer coefficient for thermal oil heat
    # exchanger. For that, first calculate Reynolds, Prandtl, and
    # Nusselt number for the salt and steam side of thermal oil charge
    # heat exchanger
    oil_hxc = m.fs.charge.thermal_oil_disjunct.hxc

    oil_hxc.oil_reynolds_number = Expression(
        expr=(
            oil_hxc.tube_inlet.flow_mass[0] *
            m.fs.charge.hxc_tube_outer_dia /
            (m.fs.charge.hxc_shell_eff_area *
             oil_hxc.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    oil_hxc.oil_prandtl_number = Expression(
        expr=(
            oil_hxc.cold_side.properties_in[0].cp_mass["Liq"]
            * oil_hxc.cold_side.properties_in[0].visc_d_phase["Liq"]
            / oil_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]),
        doc="Salt Prandtl Number")
    oil_hxc.oil_prandtl_wall = Expression(
        expr=(
            oil_hxc.cold_side.properties_out[0].cp_mass["Liq"]
            * oil_hxc.cold_side.properties_out[0].visc_d_phase["Liq"]
            / oil_hxc.cold_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Wall Prandtl Number"
    )
    oil_hxc.oil_nusselt_number = Expression(
        expr=(
            0.36 *
            ((oil_hxc.oil_reynolds_number**0.55) *
             (oil_hxc.oil_prandtl_number**0.33) *
             ((oil_hxc.oil_prandtl_number /
               oil_hxc.oil_prandtl_wall)**0.14))
        ),
        doc="Salt Nusslet Number from 2014, He et al, Exp Therm Fl Sci, 59, 9"
    )
    oil_hxc.steam_reynolds_number = Expression(
        expr=(
            oil_hxc.shell_inlet.flow_mol[0] *
            oil_hxc.hot_side.properties_in[0].mw *
            m.fs.charge.hxc_tube_inner_dia
            / (m.fs.charge.hxc_tube_cs_area
               * m.fs.charge.hxc_n_tubes
               * oil_hxc.hot_side.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number"
    )
    oil_hxc.steam_prandtl_number = Expression(
        expr=(
            (oil_hxc.hot_side.properties_in[0].cp_mol
             / oil_hxc.hot_side.properties_in[0].mw) *
            oil_hxc.hot_side.properties_in[0].visc_d_phase["Vap"]
            / oil_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number"
    )
    oil_hxc.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (oil_hxc.steam_reynolds_number ** 0.8) *
            (oil_hxc.steam_prandtl_number ** (0.33)) *
            ((oil_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
              oil_hxc.hot_side.properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Calculate heat transfer coefficient for salt and steam side of
    # charge heat exchanger
    oil_hxc.h_oil = Expression(
        expr=(
            oil_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"] *
            oil_hxc.oil_nusselt_number /
            m.fs.charge.hxc_tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    oil_hxc.h_steam = Expression(
        expr=(
            oil_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"] *
            oil_hxc.steam_nusselt_number /
            m.fs.charge.hxc_tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    # Calculate overall heat transfer coefficient for thermal oil of
    # thermal oil heat exchanger
    @m.fs.charge.thermal_oil_disjunct.hxc.Constraint(
        m.fs.time,
        doc="Thermal oil charge heat exchanger overall heat transfer coefficient")
    def constraint_hxc_ohtc_thermal_oil(b, t):
        return (
            b.overall_heat_transfer_coefficient[t] *
            (2 *
             m.fs.charge.hxc_k_steel *
             b.h_steam
             + m.fs.charge.hxc_tube_outer_dia *
             m.fs.charge.hxc_log_tube_dia_ratio *
             b.h_oil *
             b.h_steam
             + m.fs.charge.hxc_tube_dia_ratio *
             b.h_oil *
             2 * m.fs.charge.hxc_k_steel)
        ) == (2 * m.fs.charge.hxc_k_steel *
              b.h_oil *
              b.h_steam)

    # Define arc to connect units within disjunct
    m.fs.charge.thermal_oil_disjunct.connector_to_hxc = Arc(
        source=m.fs.charge.connector.outlet,
        destination=m.fs.charge.thermal_oil_disjunct.hxc.shell_inlet,
        doc="Connection from connector to thermal oil charge heat exchanger"
    )
    m.fs.charge.thermal_oil_disjunct.hxc_to_coolconnector = Arc(
        source=m.fs.charge.thermal_oil_disjunct.hxc.shell_outlet,
        destination=m.fs.charge.cooler_connector.inlet,
        doc="Connection from thermal oil charge heat exchanger to cooler connector"
    )


def vhp_source_disjunct_equations(disj):
    """Disjunction 2: selection of very high pressure steam source
    """

    m = disj.model()

    m.fs.charge.vhp_source_disjunct.ess_vhp_split = HelmSplitter(
        property_package=m.fs.prop_water,
        outlet_list=["to_hxc", "to_turbine"],
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
    """Disjunction 2: selection of high pressure source
    """

    m = disj.model()

    m.fs.charge.hp_source_disjunct.ess_hp_split = HelmSplitter(
        property_package=m.fs.prop_water,
        outlet_list=["to_hxc", "to_turbine"],
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

    # Define arcs to connect hp splitter to connector
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


def cooler_disjunct_equations(disj):
    """Disjunction 4: use a cooler
    """

    m = disj.model()

    # A cooler is added after the storage heat exchanger to ensure the
    # outlet of the charge heat exchanger is a subcooled liquid before
    # mixing it with the plant
    m.fs.charge.cooler_disjunct.cooler = Heater(
        property_package=m.fs.prop_water,
        has_pressure_change=True,
    )

    m.fs.charge.cooler_disjunct.coolconnector_to_cooler = Arc(
        source=m.fs.charge.cooler_connector.outlet,
        destination=m.fs.charge.cooler_disjunct.cooler.inlet,
    )

    m.fs.charge.cooler_disjunct.cooler_to_hxpump = Arc(
        source=m.fs.charge.cooler_disjunct.cooler.outlet,
        destination=m.fs.charge.hx_pump.inlet
    )

    # The temperature at the outlet of the cooler is required to be subcooled
    # by at least 5 degrees
    m.fs.charge.cooler_disjunct.constraint_cooler_connector_enth2 = Constraint(
        expr=m.fs.charge.cooler_disjunct.cooler.control_volume.properties_out[0].temperature <= \
        (m.fs.charge.cooler_disjunct.cooler.control_volume.properties_out[0].temperature_sat - 5)
    )

    m.fs.charge.cooler_disjunct.constraint_cooler_duty = Constraint(
        expr=m.fs.charge.cooler_heat_duty[0] == m.fs.charge.cooler_disjunct.cooler.heat_duty[0]
    )

    # Add a cost function for the cooler
    m.fs.charge.cooler_disjunct.constraint_cooler_capital_cost_function = Constraint(
        expr=m.fs.charge.cooler_capital_cost == (
            (28300
             - 0.0058 * m.fs.charge.cooler_disjunct.cooler.heat_duty[0]
             + 5e-10 * (m.fs.charge.cooler_disjunct.cooler.heat_duty[0]**2)
            ) / m.fs.charge.num_of_years
        )
    )


def no_cooler_disjunct_equations(disj):
    """Disjunction 4: no cooler
    """

    m = disj.model()

    m.fs.charge.no_cooler_disjunct.coolconnector_to_hxpump = Arc(
        source=m.fs.charge.cooler_connector.outlet,
        destination=m.fs.charge.hx_pump.inlet
    )

    m.fs.charge.no_cooler_disjunct.constraint_cooler_enth2 = Constraint(
        expr=m.fs.charge.cooler_connector.control_volume.properties_out[0].temperature <= \
        (m.fs.charge.cooler_connector.control_volume.properties_out[0].temperature_sat - 5)
    )

    # Add a constraint to ensure the cooler heat duty is equal to zero
    # since no cooler is used
    m.fs.charge.no_cooler_disjunct.constraint_cooler_duty = Constraint(
        expr=m.fs.charge.cooler_heat_duty[0] == 0
    )

    # Add a zero cost for cooler since it is not included
    m.fs.charge.no_cooler_disjunct.constraint_cooler_zero_cost = Constraint(
        expr=m.fs.charge.cooler_capital_cost == 0
    )


def recycle_mixer3_sink_disjunct_equations(disj):
    """Disjunction 4: sink disjunction
    """

    m = disj.model()

    #  Add recycle mixer
    m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_bfw_out", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    m.fs.charge.recycle_mixer3_sink_disjunct.recyclemixer_pressure_cosntraint = Constraint(
        expr=(
            m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_bfw_out_state[0].pressure == 
            m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.mixed_state[0].pressure
        ),
        doc="Recycle mixer outlet pressure equal to minimum pressure in inlets")

    m.fs.charge.recycle_mixer3_sink_disjunct.hxpump_to_recyclemix = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_hx_pump,
        doc="Connection from HX pump to recycle mixer"
    )
    m.fs.charge.recycle_mixer3_sink_disjunct.bfp_to_recyclemix = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_bfw_out,
        doc="Connection from BFP outlet to recycle mixer"
    )
    m.fs.charge.recycle_mixer3_sink_disjunct.recyclemix_to_fwh8 = Arc(
        source=m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.outlet,
        destination=m.fs.fwh[8].tube_inlet,
        doc="Connection from Recycle Mixer 3 to FWH8 tube side"
    )

    # Reconnect FWH8 outlet 2 to FWH9 inlet 2
    m.fs.charge.recycle_mixer3_sink_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].tube_outlet,
        destination=m.fs.fwh[9].tube_inlet,
        doc="Connection from Recycle Mixer 3 to FWH8 tube side"
    )

    # Reconnect FWH9 to boiler
    m.fs.charge.recycle_mixer3_sink_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet
    )

    # Reconnect FWH6 to FWH7
    m.fs.charge.recycle_mixer3_sink_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].tube_outlet,
        destination=m.fs.fwh[7].tube_inlet,
        doc="Connection from FWH6 outlet 2 to FWH7 inlet 2"
    )

    # Reconnect booster to FWH6
    m.fs.charge.recycle_mixer3_sink_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].tube_inlet
    )


def mixer1_sink_disjunct_equations(disj):
    """Disjunction 4: sink disjunction
    """

    m = disj.model()

    # Add recycle mixer 4
    m.fs.charge.recycle_mixer4_sink_disjunct.mixer1 = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_fwh8", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    m.fs.charge.recycle_mixer4_sink_disjunct.mixer1_pressure_constraint = Constraint(
        expr=m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_fwh8_state[0].pressure == \
        m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.mixed_state[0].pressure,
        doc="Recycle mixer 4 outlet pressure equal to minimum pressure in inlets")

    m.fs.charge.recycle_mixer4_sink_disjunct.hxpump_to_mix1 = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_hx_pump,
        doc="Connection from HX pump to recycle mixer 4"
    )
    m.fs.charge.recycle_mixer4_sink_disjunct.fwh8_to_mix1 = Arc(
        source=m.fs.fwh[8].tube_outlet,
        destination=m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_fwh8,
        doc="Connection from FWH8 outlet 2 to recycle mixer 4"
    )
    m.fs.charge.recycle_mixer4_sink_disjunct.mix1_to_fwh9 = Arc(
        source=m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.outlet,
        destination=m.fs.fwh[9].tube_inlet,
        doc="Connection from recycle mixer 4 to FWH9 tube side"
    )

    # Reconnect BFP to FWH8 inlet 2
    m.fs.charge.recycle_mixer4_sink_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].tube_inlet,
        doc="Connection from Recycle Mixer 3 to FWH8 tube side"
    )

    # Reconnect FWH9 to boiler
    m.fs.charge.recycle_mixer4_sink_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet
    )

    # Reconnect FWH6 to FWH7
    m.fs.charge.recycle_mixer4_sink_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].tube_outlet,
        destination=m.fs.fwh[7].tube_inlet,
        doc="Connection from FWH6 outlet 2 to FWH7 inlet 2"
    )

    # Reconnect booster to FWH6
    m.fs.charge.recycle_mixer4_sink_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].tube_inlet
    )


def mixer2_sink_disjunct_equations(disj):
    """Disjunction 4: sink disjunction
    """

    m = disj.model()

    #  Add mixer 2
    m.fs.charge.recycle_mixer5_sink_disjunct.mixer2 = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_fwh9", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    m.fs.charge.recycle_mixer5_sink_disjunct.mixer2_pressure_constraint = Constraint(
        expr=m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_fwh9_state[0].pressure == \
        m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.mixed_state[0].pressure,
        doc="mixer 2 outlet pressure equal to minimum pressure in inlets")

    m.fs.charge.recycle_mixer5_sink_disjunct.hxpump_to_mix2 = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_hx_pump,
        doc="Connection from HX pump to mixer 2"
    )
    m.fs.charge.recycle_mixer5_sink_disjunct.fwh9_to_mix2 = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_fwh9,
        doc="Connection from FWH9 outlet 2 to mixer 2"
    )
    m.fs.charge.recycle_mixer5_sink_disjunct.mix2_to_fwh9 = Arc(
        source=m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.outlet,
        destination=m.fs.boiler.inlet,
        doc="Connection from recycle mixer 4 to boiler"
    )

    # Reconnect BFP to FWH8 inlet 2
    m.fs.charge.recycle_mixer5_sink_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].tube_inlet,
        doc="Connection from Recycle Mixer 3 to FWH8 tube side"
    )

    # Reconnect FWH8 outlet 2 to FWH9 inlet 2
    m.fs.charge.recycle_mixer5_sink_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].tube_outlet,
        destination=m.fs.fwh[9].tube_inlet,
        doc="Connection from FWH8 tube side to FWH9 tube side"
    )

    # Reconnect FWH6 to FWH7
    m.fs.charge.recycle_mixer5_sink_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].tube_outlet,
        destination=m.fs.fwh[7].tube_inlet,
        doc="Connection from FWH6 outlet 2 to FWH7 inlet 2"
    )

    # Reconnect booster to FWH6
    m.fs.charge.recycle_mixer5_sink_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].tube_inlet
    )


def recycle_mixer2_sink_disjunct_equations(disj):
    """Disjunction 4: sink disjunction
    """

    m = disj.model()

    #  Add recycle mixer 2
    m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2 = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_fwh6", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2_pressure_constraint = Constraint(
        expr=m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_fwh6_state[0].pressure == \
        m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.mixed_state[0].pressure,
        doc="recycle mixer 2 outlet pressure equal to minimum pressure in inlets")

    m.fs.charge.recycle_mixer2_sink_disjunct.hxpump_to_mix3 = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_hx_pump,
        doc="Connection from HX pump to recycle mixer 2"
    )
    m.fs.charge.recycle_mixer2_sink_disjunct.fwh6_to_mix3 = Arc(
        source=m.fs.fwh[6].tube_outlet,
        destination=m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_fwh6,
        doc="Connection from FWH6 outlet 2 to recycle mixer 2"
    )
    m.fs.charge.recycle_mixer2_sink_disjunct.mix3_to_fwh7 = Arc(
        source=m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.outlet,
        destination=m.fs.fwh[7].tube_inlet,
        doc="Connection from recycle mixer 2 to FWH7"
    )

    # Reconnect BFP to FWH8 inlet 2
    m.fs.charge.recycle_mixer2_sink_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].tube_inlet,
        doc="Connection from Recycle recycle mixer 2 to FWH8 tube side"
    )

    # Reconnect FWH8 outlet 2 to FWH9 inlet 2
    m.fs.charge.recycle_mixer2_sink_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].tube_outlet,
        destination=m.fs.fwh[9].tube_inlet,
        doc="Connection from FWH8 tube side to FWH9 tube side"
    )

    # Reconnect FWH9 to boiler
    m.fs.charge.recycle_mixer2_sink_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet
    )

    # Reconnect booster to FWH6
    m.fs.charge.recycle_mixer2_sink_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].tube_inlet
    )


def recycle_mixer1_sink_disjunct_equations(disj):
    """Disjunction 4: sink disjunction
    """

    m = disj.model()

    #  Add recycle mixer 1
    m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1 = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_booster", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1_pressure_constraint = Constraint(
        expr=m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_booster_state[0].pressure == \
        m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.mixed_state[0].pressure,
        doc="recycle mixer 1 outlet pressure equal to minimum pressure in inlets")

    m.fs.charge.recycle_mixer1_sink_disjunct.hxpump_to_mix4 = Arc(
        source=m.fs.charge.hx_pump.outlet,
        destination=m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_hx_pump,
        doc="Connection from HX pump to recycle mixer 1"
    )
    m.fs.charge.recycle_mixer1_sink_disjunct.booster_to_mix4 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_booster,
        doc="Connection from booster to recycle mixer 1"
    )
    m.fs.charge.recycle_mixer1_sink_disjunct.mix4_to_fwh6 = Arc(
        source=m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.outlet,
        destination=m.fs.fwh[6].tube_inlet,
        doc="Connection from recycle mixer 1 to FWH6"
    )

    # Reconnect BFP to FWH8 inlet 2
    m.fs.charge.recycle_mixer1_sink_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].tube_inlet,
        doc="Connection from BFP to FWH8 tube side"
    )

    # Reconnect FWH8 outlet 2 to FWH9 inlet 2
    m.fs.charge.recycle_mixer1_sink_disjunct.fwh8_to_fwh9 = Arc(
        source=m.fs.fwh[8].tube_outlet,
        destination=m.fs.fwh[9].tube_inlet,
        doc="Connection from FWH8 tube side to FWH9 tube side"
    )

    # Reconnect FWH9 to boiler
    m.fs.charge.recycle_mixer1_sink_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet
    )

    # Reconnect FWH6 to FWH7
    m.fs.charge.recycle_mixer1_sink_disjunct.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh[6].tube_outlet,
        destination=m.fs.fwh[7].tube_inlet,
        doc="Connection from FWH6 outlet 2 to FWH7 inlet 2"
    )


def set_model_input(m):
    """Define model inputs and fixed variables or parameter values
    """

    # All the parameter values in this block, unless otherwise stated
    # explicitly, are either assumed or estimated for a total power
    # out of 437 MW

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    ###########################################################################
    #  Charge Heat Exchanger section                                          #
    ###########################################################################
    # Add heat exchanger area from supercritical plant model_input. For
    # conceptual design optimization, area is unfixed and optimized
    m.fs.charge.solar_salt_disjunct.hxc.area.fix(2000)  # m2
    m.fs.charge.hitec_salt_disjunct.hxc.area.fix(1200)  # m2
    m.fs.charge.thermal_oil_disjunct.hxc.area.fix(2000)  # from Andres's model

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.charge.solar_salt_disjunct.hxc.tube_inlet.flow_mass.fix(250)   # kg/s
    m.fs.charge.solar_salt_disjunct.hxc.tube_inlet.temperature.fix(513.15)  # K
    m.fs.charge.solar_salt_disjunct.hxc.tube_inlet.pressure.fix(101325)  # Pa

    m.fs.charge.hitec_salt_disjunct.hxc.tube_inlet.flow_mass.fix(400)   # kg/s
    m.fs.charge.hitec_salt_disjunct.hxc.tube_inlet.temperature.fix(435.15)  # K
    m.fs.charge.hitec_salt_disjunct.hxc.tube_inlet.pressure.fix(101325)  # Pa

    # -------- from Andres's model (Begin) --------
    # m.fs.charge.thermal_oil_disjunct.hxc.
    # overall_heat_transfer_coefficient.fix(432.677)
    m.fs.charge.thermal_oil_disjunct.hxc.tube_inlet.flow_mass[0].fix(400)
    m.fs.charge.thermal_oil_disjunct.hxc.tube_inlet.temperature[0].fix(353.15)
    m.fs.charge.thermal_oil_disjunct.hxc.tube_inlet.pressure[0].fix(101325)
    # -------- from Andres's model (End) --------

    # Cooler outlet enthalpy is fixed during model build to ensure the
    # inlet to the pump is liquid. However, this is unfixed during
    # design optimization. The temperature is at the outlet of cooler
    # is constrained in the model
    m.fs.charge.cooler_disjunct.cooler.outlet.enth_mol[0].fix(14000)
    m.fs.charge.cooler_disjunct.cooler.deltaP[0].fix(0)

    # HX pump efficiecncy assumption
    m.fs.charge.hx_pump.efficiency_pump.fix(0.80)
    m.fs.charge.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    ###########################################################################
    #  ESS VHP and HP splitters                                               #
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.charge.vhp_source_disjunct.ess_vhp_split.split_fraction[0, "to_hxc"].fix(0.15)
    m.fs.charge.hp_source_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].fix(0.15)

    ###########################################################################
    #  Connectors
    ###########################################################################
    # Fix heat duty to zero for dummy connectors
    m.fs.charge.connector.heat_duty[0].fix(0)
    m.fs.charge.cooler_connector.heat_duty[0].fix(0)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

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

    for k in [m.fs.charge.cooler_disjunct.cooler,
              m.fs.charge.connector,
              m.fs.charge.cooler_connector]:
        iscale.set_scaling_factor(k.control_volume.heat, 1e-6)


def set_var_scaling(m):
    iscale.set_scaling_factor(m.fs.charge.cooler_capital_cost, 1e-3)
    # iscale.set_scaling_factor(m.fs.charge.cooler_disjunct.constraint_cooler_capital_cost_function,
    #                           1e-3)
    iscale.set_scaling_factor(m.fs.charge.capital_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.solar_salt_disjunct.capital_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.hitec_salt_disjunct.capital_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.thermal_oil_disjunct.capital_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.operating_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.plant_capital_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.plant_fixed_operating_cost, 1e-6)
    iscale.set_scaling_factor(m.fs.charge.plant_variable_operating_cost, 1e-6)


def initialize(m, solver=None, optarg=None, outlvl=idaeslog.NOTSET):
    """Initialize the units included in the charge model
    """

    # Include scaling factors
    iscale.calculate_scaling_factors(m)

    # Initialize splitters
    propagate_state(m.fs.charge.vhp_source_disjunct.boiler_to_essvhp)
    m.fs.charge.vhp_source_disjunct.ess_vhp_split.initialize(outlvl=outlvl,
                                                             optarg=solver.options)
    propagate_state(m.fs.charge.hp_source_disjunct.rh1_to_esshp)
    m.fs.charge.hp_source_disjunct.ess_hp_split.initialize(outlvl=outlvl,
                                                           optarg=solver.options)

    # Re-initialize turbines connected to splitters since the flow is
    # not the same as before
    propagate_state(m.fs.charge.hp_source_disjunct.boiler_to_turb1)
    m.fs.turbine[1].initialize(outlvl=outlvl,
                               optarg=solver.options)
    propagate_state(m.fs.charge.hp_source_disjunct.esshp_to_turb3)
    m.fs.turbine[3].initialize(outlvl=outlvl,
                               optarg=solver.options)

    # Initialize connector
    propagate_state(m.fs.charge.hp_source_disjunct.hpsplit_to_connector)
    m.fs.charge.connector.initialize(outlvl=outlvl,
                                     optarg=solver.options)

    # Initialize solar salt, hitec salt, and thermal oil storage heat
    # exchanger by fixing the charge steam inlet during
    # initialization. Note that these should be unfixed during
    # optimization
    propagate_state(m.fs.charge.solar_salt_disjunct.connector_to_hxc)
    m.fs.charge.solar_salt_disjunct.hxc.initialize(outlvl=outlvl,
                                                   optarg=solver.options)

    propagate_state(m.fs.charge.hitec_salt_disjunct.connector_to_hxc)
    m.fs.charge.hitec_salt_disjunct.hxc.initialize(
        outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.charge.thermal_oil_disjunct.connector_to_hxc)
    m.fs.charge.thermal_oil_disjunct.hxc.initialize(outlvl=outlvl)

    # Initialize cooler connector
    propagate_state(m.fs.charge.solar_salt_disjunct.hxc_to_coolconnector)
    m.fs.charge.cooler_connector.initialize(outlvl=outlvl,
                                            optarg=solver.options)

    # Initialize cooler
    propagate_state(m.fs.charge.cooler_disjunct.coolconnector_to_cooler)
    m.fs.charge.cooler_disjunct.cooler.initialize(outlvl=outlvl,
                                                  optarg=solver.options)

    # Initialize HX pump
    propagate_state(m.fs.charge.cooler_disjunct.cooler_to_hxpump)
    m.fs.charge.hx_pump.initialize(outlvl=outlvl,
                                   optarg=solver.options)

    #  Recycle mixer initialization
    propagate_state(m.fs.charge.recycle_mixer3_sink_disjunct.bfp_to_recyclemix)
    propagate_state(m.fs.charge.recycle_mixer3_sink_disjunct.hxpump_to_recyclemix)
    m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.initialize(outlvl=outlvl)

    #  recycle mixer 4 initialization
    propagate_state(m.fs.charge.recycle_mixer4_sink_disjunct.fwh8_to_mix1)
    propagate_state(m.fs.charge.recycle_mixer4_sink_disjunct.hxpump_to_mix1)
    m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.initialize(outlvl=outlvl)

    # Recycle mixer 5 initialization
    propagate_state(m.fs.charge.recycle_mixer5_sink_disjunct.fwh9_to_mix2)
    propagate_state(m.fs.charge.recycle_mixer5_sink_disjunct.hxpump_to_mix2)
    m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.initialize(outlvl=outlvl)

    #  recycle mixer 2 initialization
    propagate_state(m.fs.charge.recycle_mixer2_sink_disjunct.fwh6_to_mix3)
    propagate_state(m.fs.charge.recycle_mixer2_sink_disjunct.hxpump_to_mix3)
    m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.initialize(outlvl=outlvl)

    #  recycle mixer 1 initialization
    propagate_state(m.fs.charge.recycle_mixer1_sink_disjunct.booster_to_mix4)
    propagate_state(m.fs.charge.recycle_mixer1_sink_disjunct.hxpump_to_mix4)
    m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.initialize(outlvl=outlvl)

    # Fix disjuncts for initialization
    m.fs.charge.solar_salt_disjunct.indicator_var.fix(True)
    m.fs.charge.hitec_salt_disjunct.indicator_var.fix(False)
    m.fs.charge.thermal_oil_disjunct.indicator_var.fix(False)

    m.fs.charge.vhp_source_disjunct.indicator_var.fix(False)
    m.fs.charge.hp_source_disjunct.indicator_var.fix(True)

    m.fs.charge.cooler_disjunct.indicator_var.fix(False)
    m.fs.charge.no_cooler_disjunct.indicator_var.fix(True)

    m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(False)
    m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(False)
    m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(True)
    m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
    m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)

    # Clone the model to transform and initialize
    # then copy the initialized variable values
    m_init = m.clone()
    m_init_var_names = [v for v in m_init.component_data_objects(Var)]
    m_orig_var_names = [v for v in m.component_data_objects(Var)]

    TransformationFactory("gdp.fix_disjuncts").apply_to(m_init)

    # Check and raise an error if the degrees of freedom are not 0
    if not degrees_of_freedom(m_init) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m_init))
            )

    init_results = solver.solve(m_init, options=optarg)
    print("Charge model initialization solver termination = ",
          init_results.solver.termination_condition)

    for v1, v2 in zip(m_init_var_names, m_orig_var_names):
        v2.value == v1.value

    print("***************   Charge Model Initialized   ********************")
    print("***************   Charge Model Initialized   ********************")



def build_costing(m, solver=None):
    """ Add cost correlations for the storage design analysis. This
    function is used to estimate the capital and operatig cost of
    integrating an energy storage system. It contains cost
    correlations to estimate the capital cost of charge heat
    exchanger, salt storage tank, molten salt pump, and salt
    inventory. Note that it does not compute the cost of the whole
    power plant.

    """

    ###########################################################################
    # Add capital cost
    # 1. Calculate charge storage material purchase cost
    # 2. Calculate charge heat exchangers cost
    # 3. Calculate charge storage material pump purchase cost
    # 4. Calculate charge storage material vessel cost
    # 5. Calculate total capital cost of charge system

    # Main assumptions
    # 1. Salt/oil life is assumed to outlast the plant life
    # 2. The economic objective is to minimize total annualized cost. So, cash
    # flows, discount rate, and NPV are not included in this study.
    ###########################################################################
    # Add capital cost: 1. Calculate storage material purchase cost
    ###########################################################################

    #  Solar salt inventory
    m.fs.charge.solar_salt_disjunct.salt_amount = Expression(
        expr=(m.fs.charge.solar_salt_disjunct.hxc.tube_inlet.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total Solar salt inventory flow in kg"
    )
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Solar salt purchase cost in $"
    )

    def solar_salt_purchase_cost_rule(b):
        return (
            b.salt_purchase_cost == (
                b.salt_amount * m.fs.charge.solar_salt_price
            ) / m.fs.charge.num_of_years
        )
    m.fs.charge.solar_salt_disjunct.salt_purchase_cost_eq = Constraint(
        rule=solar_salt_purchase_cost_rule)

    #  Hitec salt inventory
    m.fs.charge.hitec_salt_disjunct.salt_amount = Expression(
        expr=(m.fs.charge.hitec_salt_disjunct.hxc.tube_inlet.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total Hitec salt inventory flow in gal per min"
    )
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Hitec salt purchase cost in $"
    )

    def hitec_salt_purchase_cost_rule(b):
        return (
            b.salt_purchase_cost == (
                b.salt_amount * m.fs.charge.hitec_salt_price
            ) / m.fs.charge.num_of_years
        )
    m.fs.charge.hitec_salt_disjunct.salt_purchase_cost_eq = \
        Constraint(rule=hitec_salt_purchase_cost_rule)

    #  Thermal oil inventory
    m.fs.charge.thermal_oil_disjunct.oil_amount = Expression(
        expr=(m.fs.charge.thermal_oil_disjunct.hxc.tube_inlet.flow_mass[0] *
              m.fs.charge.hours_per_day * 3600),
        doc="Total Thermal oil inventory flow in kg/s"
    )
    m.fs.charge.thermal_oil_disjunct.salt_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e10),
        doc="Thermal oil purchase cost in $"
    )

    def thermal_oil_purchase_cost_rule(b):
        return (
            b.salt_purchase_cost == (
                b.oil_amount * m.fs.charge.thermal_oil_price
            ) / m.fs.charge.num_of_years
        )
    m.fs.charge.thermal_oil_disjunct.salt_purchase_cost_eq = \
        Constraint(rule=thermal_oil_purchase_cost_rule)

    # Initialize Solar and Hitec cost correlation
    for salt_disj in [m.fs.charge.solar_salt_disjunct,
                      m.fs.charge.hitec_salt_disjunct,
                      m.fs.charge.thermal_oil_disjunct]:
        calculate_variable_from_constraint(
            salt_disj.salt_purchase_cost,
            salt_disj.salt_purchase_cost_eq)

    ###########################################################################
    # Add capital cost: 2. Calculate charge heat exchangers cost
    ###########################################################################
    # Calculate and initialize Solar salt, Hitec salt, and thermal oil
    # charge heat exchangers costs, which are estimated using the
    # IDAES costing method with default options, i.e. a U-tube heat
    # exchanger, stainless steel material, and a tube length of
    # 12ft. Refer to costing documentation to change any of the
    # default options. The purchase cost of heat exchanger has to be
    # annualized when used

    m.fs.costing = SSLWCosting()

    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                      m.fs.charge.hitec_salt_disjunct.hxc,
                      m.fs.charge.thermal_oil_disjunct.hxc]:
        salt_hxc.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method=SSLWCostingData.cost_heat_exchanger,
        )

    # Calculate and initialize storage water pump cost. The purchase
    # cost has to be annualized when used
    m.fs.charge.hx_pump.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=SSLWCostingData.cost_pump,
        costing_method_arguments={
            "pump_type": PumpType.Centrifugal,
            "material_type": PumpMaterial.StainlessSteel,
            "pump_type_factor": 1.4,
            "motor_type": PumpMotorType.Open,
        },
    )

    ###########################################################################
    # Add capital cost: 3. Calculate charge storage material pump purchase cost
    ###########################################################################
    # Pumps for moving molten salts or thermal oil are not explicity
    # modeled.  To compute capital costs for these pumps the capital
    # cost expressions are added below for each heat transfer fluid
    # (Solar salt, Hitec salt, and thermal oil).  All cost expressions
    # are from the same reference as the IDAES costing framework and
    # is given below.  Seider, Seader, Lewin, Windagdo, 3rd Ed. John
    # Wiley and Sons, Chapter 22. Cost Accounting and Capital Cost
    # Estimation, Section 22.2 Cost Indexes and Capital Investment

    # ---------- Solar salt ----------
    # Calculate purchase cost of Solar salt pump
    m.fs.charge.solar_salt_disjunct.spump_Qgpm = pyo.Expression(
        expr=(m.fs.charge.solar_salt_disjunct.hxc.cold_side.properties_in[0].flow_mass *
              (264.17 * pyo.units.gallon / pyo.units.m**3) *
              (60 * pyo.units.s / pyo.units.min) /
              (m.fs.charge.solar_salt_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"])),
        doc="Conversion of solar salt flow mass to vol flow in gallons/min"
    )
    m.fs.charge.solar_salt_disjunct.dens_lbft3 = pyo.units.convert(
        m.fs.charge.solar_salt_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"],
        to_units=pyo.units.pound / pyo.units.foot**3
    )
    m.fs.charge.solar_salt_disjunct.spump_sf = pyo.Expression(
        expr=(m.fs.charge.solar_salt_disjunct.spump_Qgpm
              * (m.fs.charge.spump_head ** 0.5)),
        doc="Pump size factor"
    )

    # Expression for pump base purchase cost
    m.fs.charge.solar_salt_disjunct.pump_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_FT * m.fs.charge.spump_FM *
            exp(9.2951 -
                0.6019 * log(m.fs.charge.solar_salt_disjunct.spump_sf) +
                0.0519 * ((log(m.fs.charge.solar_salt_disjunct.spump_sf))**2))
        ),
        doc="Base purchase cost of Solar salt pump in $"
    )

    # Expression for pump efficiency
    m.fs.charge.solar_salt_disjunct.spump_np = pyo.Expression(
        expr=(-0.316 +
              0.24015 * log(m.fs.charge.solar_salt_disjunct.spump_Qgpm) -
              0.01199 * ((log(m.fs.charge.solar_salt_disjunct.spump_Qgpm))**2)),
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

    # Define a local variable for the log of motor's power consumption
    # This will help writing concise expressions for the motor
    # purchase cost
    log_motor_pc = log(m.fs.charge.solar_salt_disjunct.motor_pc)

    # Expression for motor purchase cost
    m.fs.charge.solar_salt_disjunct.motor_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(5.4866 +
                0.13141 * log_motor_pc +
                0.053255 * (log_motor_pc**2) +
                0.028628 * (log_motor_pc**3) -
                0.0035549 * (log_motor_pc**4))
        ),
        doc="Base cost of Solar Salt pump's motor in $"
    )

    # Calculate and initialize total cost of Solar salt pump
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Salt pump and motor purchase cost in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            b.spump_purchase_cost == (
                (b.pump_CP +
                 b.motor_CP) *
                (m.CE_index / 394)
            ) / m.fs.charge.num_of_years
        )
    m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost,
        m.fs.charge.solar_salt_disjunct.spump_purchase_cost_eq)

    # ---------- Hitec salt ----------
    # Calculate cost of Hitec salt pump
    m.fs.charge.hitec_salt_disjunct.spump_Qgpm = Expression(
        expr=(
            m.fs.charge.hitec_salt_disjunct.hxc.cold_side.properties_in[0].flow_mass *
            (264.17 * pyo.units.gallon / pyo.units.m**3) *
            (60 * pyo.units.s / pyo.units.min) /
            (m.fs.charge.hitec_salt_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"])
        ),
        doc="Convert salt flow mass to volumetric flow in gal per min"
    )
    m.fs.charge.hitec_salt_disjunct.dens_lbft3 = pyo.units.convert(
        m.fs.charge.hitec_salt_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"],
        to_units=pyo.units.pound / pyo.units.foot**3
    )
    m.fs.charge.hitec_salt_disjunct.spump_sf = Expression(
        expr=m.fs.charge.hitec_salt_disjunct.spump_Qgpm *
        (m.fs.charge.spump_head ** 0.5),
        doc="Pump size factor"
    )

    # Define a local variable for the log of pump's size factor
    # calculated above. This will help writing concise expressions
    log_hitec_spump_sf = log(m.fs.charge.hitec_salt_disjunct.spump_sf)

    # Expression for pump base purchase cost
    m.fs.charge.hitec_salt_disjunct.pump_CP = Expression(
        expr=(m.fs.charge.spump_FT *
              m.fs.charge.spump_FM *
              exp(9.2951 -
                  0.6019 * log_hitec_spump_sf +
                  0.0519 * (log_hitec_spump_sf**2))),
        doc="Base purchase cost of Hitec salt pump in $"
    )

    # Expression for pump efficiency
    m.fs.charge.hitec_salt_disjunct.spump_np = Expression(
        expr=(-0.316 +
              0.24015 * log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm) -
              0.01199 * ((log(m.fs.charge.hitec_salt_disjunct.spump_Qgpm))**2)),
        doc="Fractional efficiency of the pump in horsepower"
    )

    # Expression for motor power consumption
    m.fs.charge.hitec_salt_disjunct.motor_pc = Expression(
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

    
    # Define a local variable for the log of motor's power consumption
    # This will help writing concise expressions for the motor
    # purchase cost
    log_hitec_motor_pc = log(m.fs.charge.hitec_salt_disjunct.motor_pc)
    # Expression for motor base purchase cost
    m.fs.charge.hitec_salt_disjunct.motor_CP = Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(5.4866 +
                0.13141 * log_hitec_motor_pc +
                0.053255 * (log_hitec_motor_pc**2) +
                0.028628 * (log_hitec_motor_pc**3) -
                0.0035549 * (log_hitec_motor_pc**4))),
        doc="Salt Pump's Motor Base Cost in $"
    )

    # Calculate and initialize total purchase cost of Hitec salt pump
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e7),
        doc="Salt pump and motor purchase cost in $"
    )

    def hitec_spump_purchase_cost_rule(b):
        return (
            b.spump_purchase_cost == (
                (b.pump_CP +
                 b.motor_CP) *
                (m.CE_index / 394)
            ) / m.fs.charge.num_of_years
        )
    m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq = Constraint(
        rule=hitec_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost,
        m.fs.charge.hitec_salt_disjunct.spump_purchase_cost_eq)

    # ---------- Thermal oil ----------
    # Calculate cost of thermal oil pump
    m.fs.charge.thermal_oil_disjunct.spump_Qgpm = pyo.Expression(
        expr=(m.fs.charge.thermal_oil_disjunct.hxc.cold_side.properties_in[0].flow_mass *
              (264.17 * pyo.units.gallon / pyo.units.m**3) *
              (60 * pyo.units.s / pyo.units.min) /
              (m.fs.charge.thermal_oil_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"])),
        doc="Conversion of solar salt flow mass to vol flow in gallons/min"
    )
    m.fs.charge.thermal_oil_disjunct.dens_lbft3 = pyo.units.convert(
        m.fs.charge.thermal_oil_disjunct.hxc.cold_side.properties_in[0].dens_mass["Liq"],
        to_units=pyo.units.pound / pyo.units.foot**3
    )
    m.fs.charge.thermal_oil_disjunct.spump_sf = pyo.Expression(
        expr=(m.fs.charge.thermal_oil_disjunct.spump_Qgpm *
              (m.fs.charge.spump_head ** 0.5)),
        doc="Pump size factor"
    )

    # Defining a local variable for the log of pump's size factor
    # calculated above This will help writing the pump's purchase cost
    # expressions conciesly
    log_thermal_oil_spump_sf = log(m.fs.charge.thermal_oil_disjunct.spump_sf)
    # Expression for pump base purchase cost
    m.fs.charge.thermal_oil_disjunct.pump_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_FT *
            m.fs.charge.spump_FM *
            exp(9.2951 -
                0.6019 * log_thermal_oil_spump_sf +
                0.0519 * (log_thermal_oil_spump_sf**2))
        ),
        doc="Salt pump base (purchase) cost in $"
    )

    # Expression for pump efficiency
    m.fs.charge.thermal_oil_disjunct.spump_np = pyo.Expression(
        expr=(-0.316 +
              0.24015 * log(m.fs.charge.thermal_oil_disjunct.spump_Qgpm) -
              0.01199 * ((log(m.fs.charge.thermal_oil_disjunct.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump horse power"
    )

    # Expressiong for motor base purchase cost
    m.fs.charge.thermal_oil_disjunct.motor_pc = pyo.Expression(
        expr=(
            (m.fs.charge.thermal_oil_disjunct.spump_Qgpm *
             m.fs.charge.spump_head *
             m.fs.charge.thermal_oil_disjunct.dens_lbft3) /
            (33000 * m.fs.charge.thermal_oil_disjunct.spump_np *
             m.fs.charge.spump_nm)
        ),
        doc="Motor power consumption in horsepower"
    )

    # Defining a local variable for the log of motor's power
    # consumption. This will help writing the motor's purchase cost
    # expressions conciesly
    log_thermal_oil_motor_pc = log(m.fs.charge.thermal_oil_disjunct.motor_pc)
    m.fs.charge.thermal_oil_disjunct.motor_CP = pyo.Expression(
        expr=(
            m.fs.charge.spump_motorFT *
            exp(5.4866 +
                0.13141 * log_thermal_oil_motor_pc +
                0.053255 * (log_thermal_oil_motor_pc**2) +
                0.028628 * (log_thermal_oil_motor_pc**3) -
                0.0035549 * (log_thermal_oil_motor_pc**4))
        ),
        doc="Base cost of thermal oil pump's motor in $"
    )

    # Calculate and initialize total purchase cost of thermal oil
    # total pump
    m.fs.charge.thermal_oil_disjunct.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, 1e10),
        doc="Salt pump and motor purchase cost in $"
    )

    def oil_spump_purchase_cost_rule(b):
        return (
            b.spump_purchase_cost == (
                (b.pump_CP +
                 b.motor_CP) *
                (m.CE_index / 394)
            ) / m.fs.charge.num_of_years
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
    # Tank size and dimension computation
    m.fs.charge.solar_salt_disjunct.tank_volume = pyo.Var(
        initialize=1000,
        bounds=(1, 6000),
        units=pyunits.m**3,
        doc="Volume of the Salt Tank w/20% excess capacity")
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
        doc="Length of the salt tank [m]")
    m.fs.charge.solar_salt_disjunct.no_of_tanks = pyo.Param(
        initialize=1,
        doc='No of Tank units to use cost correlations')

    # Compute Solar salt tank volume with a 10% margin
    def solar_tank_volume_rule(b):
        return (
            b.tank_volume *
            b.hxc.cold_side.properties_in[0].dens_mass["Liq"] ==
            b.salt_amount *
            1.10
        )
    m.fs.charge.solar_salt_disjunct.tank_volume_eq = pyo.Constraint(rule=solar_tank_volume_rule)

    # Compute Solar salt tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def solar_tank_surf_area_rule(b):
        return (
            b.tank_surf_area == (
                pi * b.tank_diameter * b.tank_height) +
            (pi * b.tank_diameter**2) / 4
        )
    m.fs.charge.solar_salt_disjunct.tank_surf_area_eq = pyo.Constraint(rule=solar_tank_surf_area_rule)

    # Compute Solar salt tank diameter for an assumed lenght and
    # diameter
    def solar_tank_diameter_rule(b):
        return (
            b.tank_diameter == (
                (4 * (b.tank_volume / b.no_of_tanks) /
                 (m.fs.charge.l_by_d * pi)) ** (1 / 3))
        )
    m.fs.charge.solar_salt_disjunct.tank_diameter_eq = pyo.Constraint(
        rule=solar_tank_diameter_rule)

    # Compute height of Solar salt tank
    def solar_tank_height_rule(b):
        return b.tank_height == (m.fs.charge.l_by_d * b.tank_diameter)
    m.fs.charge.solar_salt_disjunct.tank_height_eq = pyo.Constraint(rule=solar_tank_height_rule)

    # Initialize tanks design correlations
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
        return b.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_dens_mass *
            m.fs.charge.solar_salt_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.solar_salt_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_tank_material_cost)

    def rule_tank_insulation_cost(b):
        return b.tank_insulation_cost == (
            m.fs.charge.storage_tank_insulation_cost *
            m.fs.charge.solar_salt_disjunct.tank_surf_area
        )

    m.fs.charge.solar_salt_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_tank_insulation_cost)

    def rule_tank_foundation_cost(b):
        return b.tank_foundation_cost == (
            m.fs.charge.storage_tank_foundation_cost * pi *
            m.fs.charge.solar_salt_disjunct.tank_diameter**2 / 4
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
        doc="Volume of the Salt Tank w/20% excess capacity")
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
        doc="Length of the salt tank [m]")
    m.fs.charge.hitec_salt_disjunct.no_of_tanks = pyo.Param(
        initialize=1,
        doc='No of Tank units to use cost correlations')

    # Computing tank volume with a 20% margin
    def hitec_tank_volume_rule(b):
        return (
            b.tank_volume * b.hxc.cold_side.properties_in[0].dens_mass["Liq"] ==
            b.salt_amount * 1.10
        )
    m.fs.charge.hitec_salt_disjunct.tank_volume_eq = Constraint(rule=hitec_tank_volume_rule)

    # Compute Hitec salt tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def hitec_tank_surf_area_rule(b):
        return b.tank_surf_area == (
            (pi * b.tank_diameter * b.tank_height) +
            (pi * b.tank_diameter**2) / 4
        )
    m.fs.charge.hitec_salt_disjunct.tank_surf_area_eq = Constraint(rule=hitec_tank_surf_area_rule)

    # Compute Hitec salt tank diameter for an assumed lenght and
    # diameter
    def hitec_tank_diameter_rule(b):
        return b.tank_diameter == (
            (4 * (b.tank_volume / b.no_of_tanks) /
             (m.fs.charge.l_by_d * pi)) ** (1 / 3)
        )
    m.fs.charge.hitec_salt_disjunct.tank_diameter_eq = Constraint(rule=hitec_tank_diameter_rule)

    # Compute Hitec salt tank height
    def hitec_tank_height_rule(b):
        return b.tank_height == (m.fs.charge.l_by_d * b.tank_diameter)
    m.fs.charge.hitec_salt_disjunct.tank_height_eq = Constraint(rule=hitec_tank_height_rule)

    # Initialize tanks design correlations
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
        return b.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_dens_mass *
            m.fs.charge.hitec_salt_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_material_cost = pyo.Constraint(
        rule=rule_hitec_tank_material_cost)

    def rule_hitec_tank_insulation_cost(b):
        return b.tank_insulation_cost == (
            m.fs.charge.storage_tank_insulation_cost *
            m.fs.charge.hitec_salt_disjunct.tank_surf_area
        )
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_insulation_cost = pyo.Constraint(
        rule=rule_hitec_tank_insulation_cost)

    def rule_hitec_tank_foundation_cost(b):
        return b.tank_foundation_cost == (
            m.fs.charge.storage_tank_foundation_cost *
            pi * m.fs.charge.hitec_salt_disjunct.tank_diameter**2 / 4
        )
    m.fs.charge.hitec_salt_disjunct.costing.eq_tank_foundation_cost = pyo.Constraint(
        rule=rule_hitec_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge.hitec_salt_disjunct.costing.total_tank_cost = Expression(
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
        doc="Volume of the Salt Tank w/20% excess capacity")
    m.fs.charge.thermal_oil_disjunct.tank_surf_area = pyo.Var(
        initialize=1000,
        bounds=(1, 6000),
        units=pyunits.m**2,
        doc="surface area of the Salt Tank")
    m.fs.charge.thermal_oil_disjunct.tank_diameter = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 40),
        units=pyunits.m,
        doc="Diameter of the Salt Tank ")
    m.fs.charge.thermal_oil_disjunct.tank_height = pyo.Var(
        initialize=1.0,
        bounds=(0.5, 13),
        units=pyunits.m,
        doc="Length of the salt tank [m]")
    m.fs.charge.thermal_oil_disjunct.no_of_tanks = pyo.Param(
        initialize=1,
        doc='No of Tank units to use cost correlations')

    # Compute thermal oil tank volume with a 10% margin
    def oil_tank_volume_rule(b):
        return (
            b.tank_volume * b.hxc.cold_side.properties_in[0].dens_mass["Liq"] ==
            b.oil_amount * 1.10
        )
    m.fs.charge.thermal_oil_disjunct.tank_volume_eq = Constraint(
        rule=oil_tank_volume_rule)

    # Compute thermal oil tank surface area considering the surface
    # area of sides and top surface area. The base area is accounted
    # in foundation costs
    def oil_tank_surf_area_rule(b):
        return b.tank_surf_area == (
            (pi * b.tank_diameter * b.tank_height) +
            (pi * b.tank_diameter**2) / 4
        )
    m.fs.charge.thermal_oil_disjunct.tank_surf_area_eq = Constraint(rule=oil_tank_surf_area_rule)

    # Compute thermal oil tank diameter for an assumed lenght and diameter
    def oil_tank_diameter_rule(b):
        return b.tank_diameter == (
            (4 * (b.tank_volume / b.no_of_tanks) /
             (m.fs.charge.l_by_d * pi)) ** (1 / 3)
        )
    m.fs.charge.thermal_oil_disjunct.tank_diameter_eq = Constraint(rule=oil_tank_diameter_rule)

    # Compute height of thermal oil tank
    def oil_tank_height_rule(b):
        return b.tank_height == (m.fs.charge.l_by_d * b.tank_diameter)
    m.fs.charge.thermal_oil_disjunct.tank_height_eq = Constraint(rule=oil_tank_height_rule)

    # Initialize tanks design correlations
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
        return b.tank_material_cost == (
            m.fs.charge.storage_tank_material_cost *
            m.fs.charge.storage_tank_material_dens_mass *
            m.fs.charge.thermal_oil_disjunct.tank_surf_area *
            m.fs.charge.tank_thickness
        )
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_material_cost = Constraint(
        rule=rule_oil_tank_material_cost)

    def rule_oil_tank_insulation_cost(b):
        return (
            b.tank_insulation_cost ==
            m.fs.charge.storage_tank_insulation_cost *
            m.fs.charge.thermal_oil_disjunct.tank_surf_area)
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_insulation_cost = Constraint(
        rule=rule_oil_tank_insulation_cost)

    def rule_oil_tank_foundation_cost(b):
        return b.tank_foundation_cost == (
            m.fs.charge.storage_tank_foundation_cost *
            pi * m.fs.charge.thermal_oil_disjunct.tank_diameter**2 / 4)
    m.fs.charge.thermal_oil_disjunct.costing.eq_tank_foundation_cost = Constraint(
        rule=rule_oil_tank_foundation_cost)

    # Expression to compute the total cost for the salt tank
    m.fs.charge.thermal_oil_disjunct.costing.total_tank_cost = Expression(
        expr=m.fs.charge.thermal_oil_disjunct.costing.tank_material_cost +
        m.fs.charge.thermal_oil_disjunct.costing.tank_foundation_cost +
        m.fs.charge.thermal_oil_disjunct.costing.tank_insulation_cost
    )

    ###########################################################################
    # Add capital cost: 5. Calculate total capital cost for charge system
    ###########################################################################
    # For the economic analysis
    # ---------- Solar salt ----------
    # Add capital cost variable at flowsheet level to handle the
    # storage material capital cost depending on the selected storage
    # material
    m.fs.charge.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e10),
        doc="Annualized capital cost")
    m.fs.charge.solar_salt_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e8),
        doc="Annualized capital cost for solar salt")

    # Annualize capital cost for the solar salt
    def solar_cap_cost_rule(b):
        return b.capital_cost == (
            b.salt_purchase_cost +
            b.spump_purchase_cost +
            (b.hxc.costing.capital_cost +
             m.fs.charge.hx_pump.costing.capital_cost +
             b.no_of_tanks *
             b.costing.total_tank_cost) /
            m.fs.charge.num_of_years
        )
    m.fs.charge.solar_salt_disjunct.cap_cost_eq = pyo.Constraint(rule=solar_cap_cost_rule)

    # Add constraint to link the global capital cost variable to
    # Solar salt capital cost from disjunction 1
    m.fs.charge.solar_salt_disjunct.fs_cap_cost_eq = Constraint(
        expr=m.fs.charge.capital_cost == m.fs.charge.solar_salt_disjunct.capital_cost)
    
    # Initialize capital cost
    calculate_variable_from_constraint(
        m.fs.charge.solar_salt_disjunct.capital_cost,
        m.fs.charge.solar_salt_disjunct.cap_cost_eq)

    # ---------- Hitec salt ----------
    m.fs.charge.hitec_salt_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e8),
        doc="Annualized capital cost for solar salt")

    # Annualize capital cost for the hitec salt
    def hitec_cap_cost_rule(b):
        return b.capital_cost == (
            b.salt_purchase_cost +
            b.spump_purchase_cost +
            (b.hxc.costing.capital_cost +
             m.fs.charge.hx_pump.costing.capital_cost +
             b.no_of_tanks *
             b.costing.total_tank_cost) /
            m.fs.charge.num_of_years
        )
    m.fs.charge.hitec_salt_disjunct.cap_cost_eq = Constraint(rule=hitec_cap_cost_rule)

    # Add constraint to link the global capital cost variable to Hitec
    # salt capital cost from disjunction 1
    m.fs.charge.hitec_salt_disjunct.fs_cap_cost_eq = Constraint(
        expr=m.fs.charge.capital_cost == m.fs.charge.hitec_salt_disjunct.capital_cost)

    # Initialize capital cost
    calculate_variable_from_constraint(
        m.fs.charge.hitec_salt_disjunct.capital_cost,
        m.fs.charge.hitec_salt_disjunct.cap_cost_eq)

    # ---------- Thermal oil ----------
    m.fs.charge.thermal_oil_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Annualized capital cost for thermal oil")

    # Annualize capital cost for the thermal oil
    def oil_cap_cost_rule(b):
        return b.capital_cost == (
            b.salt_purchase_cost +
            b.spump_purchase_cost +
            (b.hxc.costing.capital_cost +
             m.fs.charge.hx_pump.costing.capital_cost +
             b.no_of_tanks *
             b.costing.total_tank_cost) /
            m.fs.charge.num_of_years
        )
    m.fs.charge.thermal_oil_disjunct.cap_cost_eq = Constraint(rule=oil_cap_cost_rule)

    # Add constraint to link the global capital cost variable to
    # thermal oil capital cost from disjunction 1
    m.fs.charge.thermal_oil_disjunct.fs_cap_cost_eq = Constraint(
        expr=m.fs.charge.capital_cost == m.fs.charge.thermal_oil_disjunct.capital_cost)

    # Initialize capital cost
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
        doc="Operating cost")  # add units

    def op_cost_rule(b):
        return b.operating_cost == (
            b.operating_hours * m.fs.charge.coal_price *
            # (m.fs.plant_heat_duty[0] * 1e6)
            (m.fs.coal_heat_duty * 1e6)
            - (b.cooling_price * b.operating_hours *
               b.cooler_heat_duty[0])
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
    # and plant variable and fixed operating costs. Equations from
    # "USC Cost function.pptx" sent by Naresh
    m.fs.charge.plant_capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Annualized capital cost for the plant in $")
    m.fs.charge.plant_fixed_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/yr")
    m.fs.charge.plant_variable_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/yr")

    def plant_cap_cost_rule(b):
        return b.plant_capital_cost == (
            ((2688973 * m.fs.plant_power_out[0]  # in MW
              + 618968072) /
             b.num_of_years
            ) * (m.CE_index / 575.4)
        )
    m.fs.charge.plant_cap_cost_eq = Constraint(rule=plant_cap_cost_rule)

    # Initialize capital cost of power plant
    calculate_variable_from_constraint(
        m.fs.charge.plant_capital_cost,
        m.fs.charge.plant_cap_cost_eq)

    def op_fixed_plant_cost_rule(b):
        return b.plant_fixed_operating_cost == (
            ((16657.5 * m.fs.plant_power_out[0]  # in MW
              + 6109833.3) /
             b.num_of_years
            ) * (m.CE_index / 575.4)  # annualized, in $/y
        )
    m.fs.charge.op_fixed_plant_cost_eq = pyo.Constraint(
        rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return b.plant_variable_operating_cost == (
            (31754.7 * m.fs.plant_power_out[0]  # in MW
            ) * (m.CE_index / 575.4) # in $/yr
        )
    m.fs.charge.op_variable_plant_cost_eq = pyo.Constraint(
        rule=op_variable_plant_cost_rule)

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.charge.plant_fixed_operating_cost,
        m.fs.charge.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.charge.plant_variable_operating_cost,
        m.fs.charge.op_variable_plant_cost_eq)

    # Add options to GDPopt
    m_cost = m.clone()
    m_cost_var_names = [v for v in m_cost.component_data_objects(Var)]
    m_orig_var_names = [v for v in m.component_data_objects(Var)]

    TransformationFactory("gdp.fix_disjuncts").apply_to(m_cost)

    # Check and raise an error if the degrees of freedom are not 0
    if not degrees_of_freedom(m_cost) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building costing block are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m_cost))
            )


    cost_results = solver.solve(m_cost)
    print("Charge model initialization solver termination = ",
          cost_results.solver.termination_condition)

    for v1, v2 in zip(m_cost_var_names, m_orig_var_names):
        v2.value == v1.value

    print("******************** Costing Initialized *************************")
    print()
    print()
    

def unfix_disjuncts_post_initialization(m):
    """This method unfixes the disjuncts that were fixed only
    for initializing the model.

    """

    m.fs.charge.solar_salt_disjunct.indicator_var.unfix()
    m.fs.charge.hitec_salt_disjunct.indicator_var.unfix()
    m.fs.charge.thermal_oil_disjunct.indicator_var.unfix()
    m.fs.charge.vhp_source_disjunct.indicator_var.unfix()
    m.fs.charge.hp_source_disjunct.indicator_var.unfix()
    m.fs.charge.cooler_disjunct.indicator_var.unfix()
    m.fs.charge.no_cooler_disjunct.indicator_var.unfix()
    m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.unfix()
    m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.unfix()
    m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.unfix()
    m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.unfix()
    m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.unfix()

    print("******************** Disjuncts Unfixed *************************")
    print()
    print()


def calculate_bounds(m):
    m.fs.temperature_degrees = 5

    # Calculate bounds for solar salt from properties expressions
    m.fs.charge.solar_salt_temperature_max = 853.15 + m.fs.temperature_degrees # in K
    m.fs.charge.solar_salt_temperature_min = 513.15 - m.fs.temperature_degrees # in K
    m.fs.charge.solar_salt_enth_mass_max = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.charge.solar_salt_temperature_max - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.charge.solar_salt_temperature_max - 273.15)**2)
    )
    m.fs.charge.solar_salt_enth_mass_min = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.charge.solar_salt_temperature_min - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.charge.solar_salt_temperature_min - 273.15)**2)
    )

    # Calculate bounds for hitec salt from properties expressions
    m.fs.charge.hitec_salt_temperature_max = 788.15 + m.fs.temperature_degrees # in K
    m.fs.charge.hitec_salt_temperature_min = 435.15 - m.fs.temperature_degrees # in K
    m.fs.charge.hitec_salt_enth_mass_max = (
        (m.fs.hitec_salt_properties.cp_param_1.value *
         (m.fs.charge.hitec_salt_temperature_max))
        + (m.fs.hitec_salt_properties.cp_param_2.value * \
           (m.fs.charge.hitec_salt_temperature_max)**2)
        + (m.fs.hitec_salt_properties.cp_param_3.value * \
           (m.fs.charge.hitec_salt_temperature_max)**3)
    )
    m.fs.charge.hitec_salt_enth_mass_min = (
        (m.fs.hitec_salt_properties.cp_param_1.value *
         (m.fs.charge.hitec_salt_temperature_min))
        + (m.fs.hitec_salt_properties.cp_param_2.value * \
           (m.fs.charge.hitec_salt_temperature_min)**2)
        + (m.fs.hitec_salt_properties.cp_param_3.value * \
           (m.fs.charge.hitec_salt_temperature_min)**3)
    )

    # Calculate bounds for thermal oil from properties expressions
    m.fs.charge.thermal_oil_temperature_max = 616 + m.fs.temperature_degrees # in K
    # m.fs.charge.thermal_oil_temperature_min = 260 - m.fs.temperature_degrees # in K
    m.fs.charge.thermal_oil_temperature_min = 298.15 - m.fs.temperature_degrees # in K
    m.fs.charge.thermal_oil_enth_mass_max = (
        1e3 * (0.003313 * (m.fs.charge.thermal_oil_temperature_max - 273.15)**2/2 +
               0.0000008970785 * (m.fs.charge.thermal_oil_temperature_max - 273.15)**3/3 +
               1.496005 * (m.fs.charge.thermal_oil_temperature_max - 273.15))
    )
    m.fs.charge.thermal_oil_enth_mass_min = (
        1e3 * (0.003313 * (m.fs.charge.thermal_oil_temperature_min - 273.15)**2/2 +
               0.0000008970785 * (m.fs.charge.thermal_oil_temperature_min - 273.15)**3/3 +
               1.496005 * (m.fs.charge.thermal_oil_temperature_min - 273.15))
    )

    m.fs.charge.salt_enth_mass_max = max(m.fs.charge.solar_salt_enth_mass_max,
                                         m.fs.charge.hitec_salt_enth_mass_max)
    m.fs.charge.salt_enth_mass_min = min(m.fs.charge.solar_salt_enth_mass_min,
                                         m.fs.charge.hitec_salt_enth_mass_min)

    print('                         Solar        Hitec       Thermal oil')
    print('enth_mass max {: >18.4f} {: >13.4f} {: >10.4f}'.format(
        m.fs.charge.solar_salt_enth_mass_max,
        m.fs.charge.hitec_salt_enth_mass_max,
        m.fs.charge.thermal_oil_enth_mass_max))
    print('enth_mass min {: >18.4f} {: >13.4f} {: >8.4f}'.format(
        m.fs.charge.solar_salt_enth_mass_min,
        m.fs.charge.hitec_salt_enth_mass_min,
        m.fs.charge.thermal_oil_enth_mass_min))


def add_bounds(m):
    """Add bounds to units in charge model

    """

    calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W

    m.flow_max = m.main_flow * 1.2  # in mol/s
    m.storage_flow_max = m.main_flow * 0.2
    m.salt_flow_max = 1000  # in kg/s
    m.fs.heat_duty_max = 200e6 # in MW
    m.factor = 2
    # Charge heat exchanger section
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        salt_hxc.shell_inlet.flow_mol.setlb(0)
        salt_hxc.shell_inlet.flow_mol.setub(m.storage_flow_max)
        salt_hxc.tube_inlet.flow_mass.setlb(0)
        salt_hxc.tube_inlet.flow_mass.setub(m.salt_flow_max)
        salt_hxc.shell_outlet.flow_mol.setlb(0)
        salt_hxc.shell_outlet.flow_mol.setub(m.storage_flow_max)
        salt_hxc.tube_outlet.flow_mass.setlb(0)
        salt_hxc.tube_outlet.flow_mass.setub(m.salt_flow_max)
        salt_hxc.tube_inlet.pressure.setlb(101320)
        salt_hxc.tube_inlet.pressure.setub(101330)
        salt_hxc.tube_outlet.pressure.setlb(101320)
        salt_hxc.tube_outlet.pressure.setub(101330)
        salt_hxc.heat_duty.setlb(0)
        salt_hxc.heat_duty.setub(m.fs.heat_duty_max)
        salt_hxc.shell.heat.setlb(-m.fs.heat_duty_max)
        salt_hxc.shell.heat.setub(0)
        salt_hxc.tube.heat.setlb(0)
        salt_hxc.tube.heat.setub(m.fs.heat_duty_max)
        # Add calculated bounds
        salt_hxc.tube.properties_in[:].enth_mass.setlb(
            m.fs.charge.salt_enth_mass_min / m.factor)
        salt_hxc.tube.properties_in[:].enth_mass.setub(
            m.fs.charge.salt_enth_mass_max * m.factor)
        salt_hxc.tube.properties_out[:].enth_mass.setlb(
            m.fs.charge.salt_enth_mass_min / m.factor)
        salt_hxc.tube.properties_out[:].enth_mass.setub(
            m.fs.charge.salt_enth_mass_max * m.factor)
        salt_hxc.overall_heat_transfer_coefficient.setlb(0)
        salt_hxc.overall_heat_transfer_coefficient.setub(10000)
        salt_hxc.area.setlb(0)
        salt_hxc.area.setub(6000)  # TODO: Check this value
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_in.setub(88)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_out.setub(82)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_in.setlb(5)
    m.fs.charge.solar_salt_disjunct.hxc.delta_temperature_out.setlb(5)

    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_in.setub(88.2)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_out.setub(88)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_in.setlb(8)
    m.fs.charge.hitec_salt_disjunct.hxc.delta_temperature_out.setlb(8)

    for oil_hxc in [m.fs.charge.thermal_oil_disjunct.hxc]:
        oil_hxc.shell_inlet.flow_mol.setlb(0)
        oil_hxc.shell_inlet.flow_mol.setub(m.storage_flow_max)
        oil_hxc.shell_outlet.flow_mol.setlb(0)
        oil_hxc.shell_outlet.flow_mol.setub(m.storage_flow_max)
        oil_hxc.tube_inlet.flow_mass.setlb(0)
        oil_hxc.tube_inlet.flow_mass.setub(m.salt_flow_max)
        oil_hxc.tube_outlet.flow_mass.setlb(0)
        oil_hxc.tube_outlet.flow_mass.setub(m.salt_flow_max)
        oil_hxc.tube_inlet.pressure.setlb(101320)
        oil_hxc.tube_inlet.pressure.setub(101330)
        oil_hxc.tube_outlet.pressure.setlb(101320)
        oil_hxc.tube_outlet.pressure.setub(101330)
        oil_hxc.heat_duty.setlb(0)
        oil_hxc.heat_duty.setub(m.fs.heat_duty_max)
        oil_hxc.shell.heat.setlb(-m.fs.heat_duty_max)
        oil_hxc.shell.heat.setub(0)
        oil_hxc.tube.heat.setlb(0)
        oil_hxc.tube.heat.setub(m.fs.heat_duty_max)
        oil_hxc.overall_heat_transfer_coefficient.setlb(0)
        oil_hxc.overall_heat_transfer_coefficient.setub(1000)
        oil_hxc.area.setlb(0)
        oil_hxc.area.setub(8000)  # TODO: Check this value
        oil_hxc.delta_temperature_in.setub(456)
        oil_hxc.delta_temperature_out.setub(222)
        oil_hxc.delta_temperature_in.setlb(10)
        oil_hxc.delta_temperature_out.setlb(9)
        # Add calculated bounds
        oil_hxc.tube.properties_in[:].enth_mass.setlb(
            m.fs.charge.thermal_oil_enth_mass_min / m.factor)
        oil_hxc.tube.properties_in[:].enth_mass.setub(
            m.fs.charge.thermal_oil_enth_mass_max * m.factor)
        oil_hxc.tube.properties_out[:].enth_mass.setlb(
            m.fs.charge.thermal_oil_enth_mass_min / m.factor)
        oil_hxc.tube.properties_out[:].enth_mass.setub(
            m.fs.charge.thermal_oil_enth_mass_max * m.factor)

    # Add bounds for the HX pump and Cooler
    for unit_k in [m.fs.charge.connector,
                   m.fs.charge.hx_pump,
                   m.fs.charge.cooler_disjunct.cooler,
                   m.fs.charge.cooler_connector]:
        unit_k.inlet.flow_mol.setlb(0)
        unit_k.inlet.flow_mol.setub(m.storage_flow_max)
        unit_k.outlet.flow_mol.setlb(0)
        unit_k.outlet.flow_mol.setub(m.storage_flow_max)
    # m.fs.charge.cooler_disjunct.cooler.heat_duty.setlb(-1e9) # from Andres's model
    m.fs.charge.cooler_disjunct.cooler.heat_duty.setub(0)

    m.fs.charge.cooler_disjunct.cooler.deltaP.setlb(-1e10)
    m.fs.charge.cooler_disjunct.cooler.deltaP.setub(1e10)
    m.fs.charge.cooler_disjunct.cooler.heat_duty.setlb(-1e12)
    m.fs.charge.cooler_disjunct.cooler.heat_duty.setub(0)

    # Add bounds needed in VHP and HP source disjuncts
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

    for mix in [m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer,
                m.fs.charge.recycle_mixer4_sink_disjunct.mixer1,
                m.fs.charge.recycle_mixer5_sink_disjunct.mixer2,
                m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2,
                m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1]:
        mix.from_hx_pump.flow_mol.setlb(0)
        mix.from_hx_pump.flow_mol.setub(m.storage_flow_max)
        mix.outlet.flow_mol.setlb(0)
        mix.outlet.flow_mol.setub(m.flow_max)
    m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_bfw_out.flow_mol.setlb(0)
    m.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_bfw_out.flow_mol.setub(m.flow_max)
    m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_fwh8.flow_mol.setlb(0)
    m.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_fwh8.flow_mol.setub(m.flow_max)
    m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_fwh9.flow_mol.setlb(0)
    m.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_fwh9.flow_mol.setub(m.flow_max)
    m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_fwh6.flow_mol.setlb(0)
    m.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_fwh6.flow_mol.setub(m.flow_max)
    m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_booster.flow_mol.setlb(0)
    m.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_booster.flow_mol.setub(m.flow_max)

    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)
    m.fs.charge.hx_pump.control_volume.work[0].setlb(0)
    m.fs.charge.hx_pump.control_volume.work[0].setub(1e10)

    # m.fs.plant_power_out[0].setlb(300)
    # m.fs.plant_power_out[0].setub(700)

    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s

    # Adding bounds on turbine splitters flow
    for k in m.set_turbine_splitter:
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setlb(0)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setub(m.flow_max)

    return m


def add_bounds_costing(m):
    """Add bounds to all units in charge model

    """

    # Add bounds to Solar and Hitec salt charge heat exchangers
    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc]:
        salt_hxc.costing.pressure_factor.setlb(0)
        salt_hxc.costing.pressure_factor.setub(1e5)
        salt_hxc.costing.capital_cost.setlb(0)
        salt_hxc.costing.capital_cost.setub(1e7)
        salt_hxc.costing.base_cost_per_unit.setlb(0)
        salt_hxc.costing.base_cost_per_unit.setub(1e6)
        salt_hxc.costing.material_factor.setlb(0)
        salt_hxc.costing.material_factor.setub(10)

    # Add bounds to thermal oil charge heat exchanger
    for oil_hxc in [m.fs.charge.thermal_oil_disjunct.hxc]:
        # Bounds for costing
        oil_hxc.costing.pressure_factor.setlb(0)
        oil_hxc.costing.pressure_factor.setub(1e5)
        oil_hxc.costing.capital_cost.setlb(0)
        oil_hxc.costing.capital_cost.setub(1e7)
        oil_hxc.costing.base_cost_per_unit.setlb(0)
        oil_hxc.costing.base_cost_per_unit.setub(1e6)
        oil_hxc.costing.material_factor.setlb(0)
        oil_hxc.costing.material_factor.setub(10)

    # Add bounds to cost-related terms
    m.fs.charge.hx_pump.costing.capital_cost.setlb(0)
    m.fs.charge.hx_pump.costing.capital_cost.setub(1e7)

def main(m_usc, method=None, max_power=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_charge_model(m_usc, method=method, max_power=max_power)

    # Give all the required inputs to the model
    set_model_input(m)

    # Add disjunctions
    add_disjunction(m)

    # Add scaling factor
    set_scaling_factors(m)

    # Initialize the model with a sequential initialization and custom
    # routines
    print('DOF before initialization: ', degrees_of_freedom(m))
    initialize(m, solver=solver, optarg=optarg)
    print('DOF after initialization: ', degrees_of_freedom(m))

    # Add bounds
    add_bounds(m)

    # Add cost correlations
    build_costing(m, solver=solver)

    # Unfix disjuncts
    unfix_disjuncts_post_initialization(m)


    # Add bounds
    add_bounds_costing(m)

    return m, solver


def run_nlps(m,
             solver=None,
             fluid=None,
             source=None,
             sink=None,
             cooler=None):
    """This function fixes the indicator variables of the disjuncts so to
    solve NLP problems

    """

    # Disjunction 1 for the storage fluid selection
    if fluid == "solar_salt":
        m.fs.charge.solar_salt_disjunct.indicator_var.fix(True)
        m.fs.charge.hitec_salt_disjunct.indicator_var.fix(False)
        m.fs.charge.thermal_oil_disjunct.indicator_var.fix(False)
    elif fluid == "hitec_salt":
        m.fs.charge.solar_salt_disjunct.indicator_var.fix(False)
        m.fs.charge.hitec_salt_disjunct.indicator_var.fix(True)
        m.fs.charge.thermal_oil_disjunct.indicator_var.fix(False)
    elif fluid == "thermal_oil":
        m.fs.charge.solar_salt_disjunct.indicator_var.fix(False)
        m.fs.charge.hitec_salt_disjunct.indicator_var.fix(False)
        m.fs.charge.thermal_oil_disjunct.indicator_var.fix(True)
    else:
        print('Unrecognized storage fluid name!')

    # Disjunction 2 for the steam source selection
    if source == "vhp":
        m.fs.charge.vhp_source_disjunct.indicator_var.fix(True)
        m.fs.charge.hp_source_disjunct.indicator_var.fix(False)
    elif source == "hp":
        m.fs.charge.vhp_source_disjunct.indicator_var.fix(False)
        m.fs.charge.hp_source_disjunct.indicator_var.fix(True)
    else:
        print('Unrecognized source unit name!')

    # Disjunction 3 for the sink selection
    if sink == "recycle_mix":
        m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(True)
        m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)
    elif sink == "mixer1":
        m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(True)
        m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)
    elif sink == "mixer2":
        m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(True)
        m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)
    elif sink == "recycle_mixer2":
        m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(True)
        m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)
    elif sink == "recycle_mixer1":
        m.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.fix(True)
        m.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.fix(False)
        m.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.fix(False)
    else:
        print('Unrecognized sink name!')

    if cooler:
        m.fs.charge.cooler_disjunct.indicator_var.fix(True)
        m.fs.charge.no_cooler_disjunct.indicator_var.fix(False)
    else:
        m.fs.charge.cooler_disjunct.indicator_var.fix(False)
        m.fs.charge.no_cooler_disjunct.indicator_var.fix(True)

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    print("The degrees of freedom after gdp transformation ",
          degrees_of_freedom(m))

    results = solver.solve(
        m,
        tee=True,
        symbolic_solver_labels=True,
        options={
            "linear_solver": "ma27",
            "max_iter": 150
        }
    )

    return m, results


def print_model(solver_obj, nlp_model, nlp_data, csvfile):

    m_iter = solver_obj.iteration
    nlp_model.disjunction1_selection = {}
    nlp_model.disjunction2_selection = {}
    nlp_model.disjunction3_selection = {}
    nlp_model.disjunction4_selection = {}
    nlp_model.area = {}
    nlp_model.hot_salt_temp = {}
    nlp_model.storage_material_amount = {}
    nlp_model.storage_material_flow = {}
    nlp_model.steam_flow_to_storage = {}
    nlp_model.boiler_eff = {}
    nlp_model.cycle_eff = {}
    print('       ___________________________________________')
    if nlp_model.fs.charge.solar_salt_disjunct.indicator_var.value == 1:
        material_disj = nlp_model.fs.charge.solar_salt_disjunct
        nlp_model.disjunction1_selection[m_iter] = 'Solar salt is selected'
        nlp_model.storage_material_amount[m_iter] = pyo.value(material_disj.salt_amount)/1e3 #in metric ton
        print('        Disjunction 1: Solar salt is selected')
        print('          Salt amount (metric ton): {:.4f}'.format(
            pyo.value(material_disj.salt_amount)/1e3))
    elif nlp_model.fs.charge.hitec_salt_disjunct.indicator_var.value == 1:
        material_disj = nlp_model.fs.charge.hitec_salt_disjunct
        nlp_model.disjunction1_selection[m_iter] = 'Hitec salt is selected'
        nlp_model.storage_material_amount[m_iter] = pyo.value(material_disj.salt_amount)/1e3 #in metric ton
        print('        Disjunction 1: Hitec salt is selected')
        print('          Salt amount (metric ton): {:.4f}'.format(
            pyo.value(material_disj.salt_amount)/1e3))
    elif nlp_model.fs.charge.thermal_oil_disjunct.indicator_var.value == 1:
        material_disj = nlp_model.fs.charge.thermal_oil_disjunct
        nlp_model.disjunction1_selection[m_iter] = 'Thermal oil is selected'
        nlp_model.storage_material_amount[m_iter] = pyo.value(material_disj.oil_amount)/1e3 #in metric ton
        print('        Disjunction 1: Thermal oil is selected')
        print('          Salt amount (metric ton): {:.4f}'.format(
        pyo.value(material_disj.oil_amount)/1e3))    
    else:
        print('No more options!')
        
    nlp_model.area[m_iter] = pyo.value(material_disj.hxc.area)
    nlp_model.hot_salt_temp[m_iter] = pyo.value(material_disj.hxc.tube_outlet.temperature[0])
    print('          No. of tanks: {:.0f}'.format(pyo.value(material_disj.no_of_tanks)))
    print('          Delta temperature at inlet (K): {:.4f}'.format(
        pyo.value(material_disj.hxc.delta_temperature_in[0])))
    print('          Delta temperature at outlet (K): {:.4f}'.format(
        pyo.value(material_disj.hxc.delta_temperature_out[0])))
    print('          Area (m2): {:.4f}'.format(
        pyo.value(material_disj.hxc.area)))
    print('          Heat duty (MW): {:.4f}'.format(
        pyo.value(material_disj.hxc.heat_duty[0]) * 1e-6))
    print('          Cost ($/y): {:.4f}'.format(
        pyo.value(material_disj.hxc.costing.capital_cost / nlp_model.fs.charge.num_of_years)))
    nlp_model.steam_flow_to_storage[m_iter] = pyo.value(material_disj.hxc.shell_inlet.flow_mol[0])
    print('          Steam flow to storage (mol/s): {:.4f}'.format(
        pyo.value(material_disj.hxc.shell_inlet.flow_mol[0])))
    nlp_model.storage_material_flow[m_iter] = pyo.value(material_disj.hxc.tube_inlet.flow_mass[0])
    print('          Salt flow (kg/s): {:.4f}'.format(
        pyo.value(material_disj.hxc.tube_inlet.flow_mass[0])))
    print('          Water temperature in/out (K): {:.4f}/{:.4f}'.format(
        pyo.value(material_disj.hxc.hot_side.properties_in[0].temperature),
        pyo.value(material_disj.hxc.hot_side.properties_out[0].temperature)))
    print('          Salt temperature in/out (K): {:.4f}/{:.4f}'.format(
        pyo.value(material_disj.hxc.tube_inlet.temperature[0]),
        pyo.value(material_disj.hxc.tube_outlet.temperature[0])))
    print('          Heat exchanger OHTC: {:.4f}'.format(
        pyo.value(material_disj.hxc.overall_heat_transfer_coefficient[0])))

    if nlp_model.fs.charge.vhp_source_disjunct.indicator_var.value == 1:
        nlp_model.disjunction2_selection[m_iter] = 'VHP source is selected'
        print('        Disjunction 2: VHP source is selected')
        print('          ESS VHP split fraction to hxc:',
              pyo.value(nlp_model.fs.charge.vhp_source_disjunct.ess_vhp_split.split_fraction[0, "to_hxc"]))
    else:
        nlp_model.disjunction2_selection[m_iter] = 'HP is selected'
        print('        Disjunction 2: HP source is selected')
        print('          ESS HP split fraction to hxc:',
              pyo.value(nlp_model.fs.charge.hp_source_disjunct.ess_hp_split.split_fraction[0, "to_hxc"]))

    if nlp_model.fs.charge.cooler_disjunct.indicator_var.value == 1:
        nlp_model.disjunction3_selection[m_iter] = 'Cooler is selected'
        print('        Disjunction 3: Cooler is selected')
        print('          Cooler capital cost ($/yr):',
              pyo.value(nlp_model.fs.charge.cooler_capital_cost))
        print('          Cooler heat duty (MW):',
              nlp_model.fs.charge.cooler_disjunct.cooler.heat_duty[0].value * 1e-6)
        print('          Cooler Tout: ',
              pyo.value(nlp_model.fs.charge.cooler_disjunct.cooler.control_volume.properties_out[0].temperature))
        print('          Cooler Tsat: ',
              pyo.value(nlp_model.fs.charge.cooler_disjunct.cooler.control_volume.properties_out[0].temperature_sat))
    if nlp_model.fs.charge.no_cooler_disjunct.indicator_var.value == 1:
        nlp_model.disjunction3_selection[m_iter] = 'No cooler is selected'
        print('        Disjunction 3: No cooler is selected')
        print('          Cooler heat duty (MW):',
              nlp_model.fs.charge.cooler_heat_duty[0].value * 1e-6)

    if nlp_model.fs.charge.recycle_mixer1_sink_disjunct.indicator_var.value == 1:
        nlp_model.disjunction4_selection[m_iter] = 'Recycle mixer 1 is selected'
        print('        Disjunction 4: Recycle mixer 1 is selected')
        print('          HX pump outlet flow:',
              pyo.value(nlp_model.fs.charge.hx_pump.outlet.flow_mol[0]))
        print('          Booster outlet flow:',
              pyo.value(nlp_model.fs.booster.outlet.flow_mol[0]))
        print('          Recycle mixer 1 inlet from Booster:',
              pyo.value(nlp_model.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_booster.flow_mol[0]))
        print('          Recycle mixer 1 inlet from HX pump:',
              pyo.value(nlp_model.fs.charge.recycle_mixer1_sink_disjunct.recycle_mixer1.from_hx_pump.flow_mol[0]))
    elif nlp_model.fs.charge.recycle_mixer2_sink_disjunct.indicator_var.value == 1:
        nlp_model.disjunction4_selection[m_iter] = 'Recycle mixer 2 is selected'
        print('        Disjunction 4: Recycle mixer 2 is selected')
        print('          BFW outlet flow:', pyo.value(nlp_model.fs.bfp.outlet.flow_mol[0]))
        print('          FWH8 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[8].tube_inlet.flow_mol[0]))
        print('          FWH9 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[9].tube_inlet.flow_mol[0]))
        print('          HX pump outlet flow:',
              pyo.value(nlp_model.fs.charge.hx_pump.outlet.flow_mol[0]))
        print('          FWH6 outlet 2 flow:',
              pyo.value(nlp_model.fs.fwh[6].tube_outlet.flow_mol[0]))
        print('          Recycle mixer 2 inlet from FWH6:',
              pyo.value(nlp_model.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_fwh6.flow_mol[0]))
        print('          Recycle mixer 2 inlet from HX pump:',
              pyo.value(nlp_model.fs.charge.recycle_mixer2_sink_disjunct.recycle_mixer2.from_hx_pump.flow_mol[0]))
        print('          Boiler inlet flow:', pyo.value(nlp_model.fs.boiler.inlet.flow_mol[0]))
        print('          FWH9 inlet 1 flow:', pyo.value(nlp_model.fs.fwh[9].shell_inlet.flow_mol[0]))
        print('          FWH6 outlet 2 flow(mol/s)/temp(K)/press(MPa): {:.4f}/{:.4f}/{:.4f}'.format(
            pyo.value(nlp_model.fs.fwh[6].tube.properties_out[0].flow_mol),
            pyo.value(nlp_model.fs.fwh[6].tube.properties_out[0].temperature),
            pyo.value(nlp_model.fs.fwh[6].tube.properties_out[0].pressure)/1e6))
    elif nlp_model.fs.charge.recycle_mixer3_sink_disjunct.indicator_var.value == 1:
        nlp_model.disjunction4_selection[m_iter] = 'Recycle mixer 3 is selected'
        print('        Disjunction 4: Recycle mixer 3 is selected')
        print('          BFW outlet flow:', pyo.value(nlp_model.fs.bfp.outlet.flow_mol[0]))
        print('          FWH8 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[8].tube_inlet.flow_mol[0]))
        print('          FWH9 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[9].tube_inlet.flow_mol[0]))
        print('          HX pump outlet flow:',
              pyo.value(nlp_model.fs.charge.hx_pump.outlet.flow_mol[0]))
        print('          Recycle mixer 3 inlet from BFW:',
              pyo.value(nlp_model.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_bfw_out.flow_mol[0]))
        print('          Recycle mixer 3 inlet from hx pump:',
              pyo.value(nlp_model.fs.charge.recycle_mixer3_sink_disjunct.recycle_mixer.from_hx_pump.flow_mol[0]))
        print('          BFP outlet flow(mol/s)/temp(K)/press(MPa): {:.4f}/{:.4f}/{:.4f}'.format(
            pyo.value(nlp_model.fs.bfp.control_volume.properties_out[0].flow_mol),
            pyo.value(nlp_model.fs.bfp.control_volume.properties_out[0].temperature),
            pyo.value(nlp_model.fs.bfp.control_volume.properties_out[0].pressure)/1e6))
    elif nlp_model.fs.charge.recycle_mixer4_sink_disjunct.indicator_var.value == 1:
        nlp_model.disjunction4_selection[m_iter] = 'Recycle mixer 4 is selected'
        print('        Disjunction 4: Recycle mixer 4 is selected')
        print('          BFW outlet flow:', pyo.value(nlp_model.fs.bfp.outlet.flow_mol[0]))
        print('          FWH8 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[8].tube_inlet.flow_mol[0]))
        print('          HX pump outlet flow:',
              pyo.value(nlp_model.fs.charge.hx_pump.outlet.flow_mol[0]))
        print('          Recycle mixer 4 inlet from FWH8:',
              pyo.value(nlp_model.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_fwh8.flow_mol[0]))
        print('          Recycle mixer 4 inlet from HX pump:',
              pyo.value(nlp_model.fs.charge.recycle_mixer4_sink_disjunct.mixer1.from_hx_pump.flow_mol[0]))
        print('          FWH9 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[9].tube_inlet.flow_mol[0]))
        print('          FWH8 outlet 2 flow(mol/s)/temp(K)/press(MPa): {:.4f}/{:.4f}/{:.4f}'.format(
            pyo.value(nlp_model.fs.fwh[8].tube.properties_out[0].flow_mol),
            pyo.value(nlp_model.fs.fwh[8].tube.properties_out[0].temperature),
            pyo.value(nlp_model.fs.fwh[8].tube.properties_out[0].pressure)/1e6))
    elif nlp_model.fs.charge.recycle_mixer5_sink_disjunct.indicator_var.value == 1:
        nlp_model.disjunction4_selection[m_iter] = 'Recycle mixer 5 is selected'
        print('        Disjunction 4: Recycle Mixer 5 is selected')
        print('          BFW outlet flow:', pyo.value(nlp_model.fs.bfp.outlet.flow_mol[0]))
        print('          FWH8 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[8].tube_inlet.flow_mol[0]))
        print('          FWH9 inlet 2 flow:', pyo.value(nlp_model.fs.fwh[9].tube_inlet.flow_mol[0]))
        print('          HX pump outlet flow:',
              pyo.value(nlp_model.fs.charge.hx_pump.outlet.flow_mol[0]))
        print('          Recycle Mixer 5 inlet from FWH9:',
              pyo.value(nlp_model.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_fwh9.flow_mol[0]))
        print('          Recycle Mixer 5 inlet from HX pump:',
              pyo.value(nlp_model.fs.charge.recycle_mixer5_sink_disjunct.mixer2.from_hx_pump.flow_mol[0]))
        print('          Boiler inlet flow:', pyo.value(nlp_model.fs.boiler.inlet.flow_mol[0]))
        print('          FWH9 outlet 2 flow(mol/s)/temp(K)/press(MPa): {:.4f}/{:.4f}/{:.4f}'.format(
            pyo.value(nlp_model.fs.fwh[9].tube.properties_out[0].flow_mol),
            pyo.value(nlp_model.fs.fwh[9].tube.properties_out[0].temperature),
            pyo.value(nlp_model.fs.fwh[9].tube.properties_out[0].pressure)/1e6))
    else:
        print('         No other sink alternatives!')

    print()
    # for k in nlp_model.set_turbine:
    #     # nlp_model.fs.turbine[k].display()
    #     print('        Turbine {} work (MW): {:.4f}'.
    #           format(k, pyo.value(nlp_model.fs.turbine[k].work_mechanical[0]) * 1e-6))
    print('         Boiler flow inlet (mol/s): {:.4f}'.format(
        pyo.value(nlp_model.fs.boiler.inlet.flow_mol[0])))    
    nlp_model.boiler_eff[m_iter] = pyo.value(nlp_model.fs.boiler_efficiency) * 100
    print('         Boiler efficiency (%): {:.4f}'.format(
        pyo.value(nlp_model.fs.boiler_efficiency) * 100))
    nlp_model.cycle_eff[m_iter] = pyo.value(nlp_model.fs.cycle_efficiency) * 100
    print('         Cycle efficiency (%): {:.4f}'.format(
        pyo.value(nlp_model.fs.cycle_efficiency) * 100))
    print('         Plant power (MW): {:.4f}'.format(
        pyo.value(nlp_model.fs.plant_power_out[0])))
    print('         Boiler outlet conditions: Temp(K)/Press(MPa) {:.4f}/{:.4f}'.format(
        pyo.value(nlp_model.fs.boiler.control_volume.properties_out[0].temperature),
        pyo.value(nlp_model.fs.boiler.control_volume.properties_out[0].pressure)/1e6))
    print('         Reheater 1 outlet conditions: Temp(K)/Press(MPa) {:.4f}/{:.4f}'.format(
        pyo.value(nlp_model.fs.reheater[1].control_volume.properties_out[0].temperature),
        pyo.value(nlp_model.fs.reheater[1].control_volume.properties_out[0].pressure)/1e6))
    print('         HX pump outlet conditions: Temp(K)/Press(MPa) {:.4f}/{:.4f}'.format(
        pyo.value(nlp_model.fs.charge.hx_pump.control_volume.properties_out[0].temperature),
        pyo.value(nlp_model.fs.charge.hx_pump.control_volume.properties_out[0].pressure)/1e6))
    for k in nlp_model.set_turbine_splitter:
        print("         Turbine splitter {} split fraction 2: {:.4f}".
              format(k,
                     pyo.value(nlp_model.fs.turbine_splitter[k].split_fraction[0, "outlet_2"])))
    print('       ___________________________________________')
    print('')

    # Save results in dictionaries
    nlp_model.objective_value = {}
    nlp_model.objective_value[m_iter] = pyo.value(nlp_model.obj) / scaling_obj

    nlp_model.cooler_heat_duty = {}
    nlp_model.cooler_heat_duty[m_iter] = pyo.value(nlp_model.fs.charge.cooler_heat_duty[0]) * 1e-6 # MW

    if True:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                m_iter,
                nlp_model.disjunction1_selection[m_iter],
                nlp_model.disjunction2_selection[m_iter],
                nlp_model.disjunction3_selection[m_iter],
                nlp_model.disjunction4_selection[m_iter],
                nlp_model.cooler_heat_duty[m_iter],
                nlp_model.objective_value[m_iter],
                nlp_model.area[m_iter],
                nlp_model.hot_salt_temp[m_iter],
                nlp_model.storage_material_amount[m_iter],
                nlp_model.storage_material_flow[m_iter],
                nlp_model.steam_flow_to_storage[m_iter],
                nlp_model.boiler_eff[m_iter],
                nlp_model.cycle_eff[m_iter]
            ),
        )
        csvfile.flush()


def create_csv_header():
    csvfile = open('results/subnlp_master_iterations_charge_4-12disj_results.csv',
                   'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Iteration', 'Disjunction 1 (Salt selection)', 'Disjunction 2 (Source)',
         'Disjunction 3 (Cooler selection)', 'Disjunction 4 (Return selection)',
         'Cooler Heat Duty (MW)', 'Obj (MW)', 'HXArea(m2)', 'HotSaltTemp(K)',
         'StorageMaterialAmount(metric_ton)', 'StorageMaterialFlow(kg/s)',
         'SteamFlowToStorage(mol/s)', 'BoilerEff(%)', 'CycleEff(%)')
    )
    return csvfile


def run_gdp(m):
    """Declare solver GDPopt and its options
    """

    csvfile = create_csv_header()

    opt = SolverFactory('gdpopt')
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None


    results = opt.solve(
        m,
        tee=True,
        algorithm='RIC',
        mip_solver='gurobi',
        nlp_solver='ipopt',
        # OA_penalty_factor=1e4,
        # max_slack=1e4,
        init_algorithm="no_init",
        # subproblem_presolve=False,
        time_limit="2400",
        iterlim=200,
        call_after_subproblem_solve=(lambda c, a, b: print_model(c, a, b, csvfile)),
        nlp_solver_args=dict(
            tee=True,
            symbolic_solver_labels=True,
            options={
                "linear_solver": "ma27",
                # "tol": 1e-6,
                "max_iter": 200
            }
        )
    )
    csvfile.close()
    return results


def print_results(m, results):

    print('================================')
    print("***************** Printing Results ******************")
    print('')
    print("Disjunctions")
    for d in m.component_data_objects(ctype=Disjunct,
                                      active=True,
                                      sort=True, descend_into=True):
        if abs(d.indicator_var.value - 1) < 1e-6:
            print(d.name, ' should be selected!')
    print('')
    print('Objective ($/hr): {:.4f}'.format((pyo.value(m.obj) / scaling_obj)))
    print('salt amount (kg): {:.4f}'.format(
        pyo.value(m.fs.charge.solar_salt_disjunct.salt_amount)))
    print('Plant capital cost ($/yr): {:.4f}'.format(
        pyo.value(m.fs.charge.plant_capital_cost)))
    print('Plant fixed operating costs ($/yr): {:.4f}'.format(
        pyo.value(m.fs.charge.plant_fixed_operating_cost)))
    print('Plant variable operating costs ($/yr): {:.4f}'.format(
        pyo.value(m.fs.charge.plant_variable_operating_cost)))
    print('Charge capital cost ($/yr): {:.4f}'.format(pyo.value(m.fs.charge.capital_cost)))
    print('Charge operating costs ($/yr): {:.4f}'.format(pyo.value(m.fs.charge.operating_cost)))
    print('Plant Power (MW): {:.4f}'.format(pyo.value(m.fs.plant_power_out[0])))
    print('Boiler/cycle efficiency (%): {:.4f}/{:.4f}'.format(
        pyo.value(m.fs.boiler_efficiency) * 100,
        pyo.value(m.fs.cycle_efficiency) * 100))
    print('Boiler feed water flow (mol/s): {:.4f}'.format(pyo.value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.4f}'.format(
        pyo.value((m.fs.boiler.heat_duty[0] +
                   m.fs.reheater[1].heat_duty[0] +
                   m.fs.reheater[2].heat_duty[0]) * 1e-6)))
    print('Cooling duty (MW_th): {:.4f}'.format(pyo.value(m.fs.charge.cooler_heat_duty[0]) * -1e-6))
    print('')
    if m.fs.charge.solar_salt_disjunct.indicator_var.value == 1:
        print('Solar heat exchanger is selected!')
        material_disj = m.fs.charge.solar_salt_disjunct
        hx = m.fs.charge.solar_salt_disjunct.hxc
    elif m.fs.charge.hitec_salt_disjunct.indicator_var.value == 1:
        print('Hitec heat exchanger is selected')
        material_disj = m.fs.charge.hitec_salt_disjunct
        hx = m.fs.charge.hitec_salt_disjunct.hxc
    else:
        print('Thermal oil heat exchanger is selected')
        material_disj = m.fs.charge.thermal_oil_disjunct
        hx = m.fs.charge.thermal_oil_disjunct.hxc

    print('Charge heat exhanger:')
    print(' Area (m2): {:.4f}'.format(
        pyo.value(hx.area)))
    print(' Heat duty (MW): {:.4f}'.format(pyo.value(hx.heat_duty[0]) * 1e-6))
    print(' Cost ($/y): {:.4f}'.format(
        pyo.value(hx.costing.capital_cost / m.fs.charge.num_of_years)))
    print(' Steam flow to storage (mol/s): {:.4f}'.format(
        pyo.value(hx.shell_inlet.flow_mol[0])))
    print(' Salt flow (kg/s): {:.4f}'.format(pyo.value(hx.tube_inlet.flow_mass[0])))
    print(' Water temperature in/out (K): {:.4f}/{:.4f}'.format(
        pyo.value(hx.hot_side.properties_in[0].temperature),
        pyo.value(hx.hot_side.properties_out[0].temperature)))
    print(' Salt temperature in/out (K): {:.4f}/{:.4f}'.format(
        pyo.value(hx.tube_inlet.temperature[0]),
        pyo.value(hx.tube_outlet.temperature[0])))
    print(' Overall heat transfer coefficient: {:.4f}'.format(
        pyo.value(hx.overall_heat_transfer_coefficient[0])))
    print(' Delta temperature in/out (K): {:.4f}/{:.4f}'.format(
        pyo.value(hx.delta_temperature_in[0]),
        pyo.value(hx.delta_temperature_out[0])))
    print(' Salt cost ($/y): {:.4f}'.format(
        pyo.value(material_disj.salt_purchase_cost)))
    print(' Tank cost ($/y): {:.4f}'.format(
        pyo.value(material_disj.costing.total_tank_cost / m.fs.charge.num_of_years)))
    print(' Salt pump cost ($/y): {:.4f}'.format(
        pyo.value(material_disj.spump_purchase_cost)))
    print(' Salt storage tank volume in m3: {:.4f}'.format(
        pyo.value(material_disj.tank_volume)))
    print(' Salt dens_mass: {:.4f}'.format(
        pyo.value(hx.cold_side.properties_in[0].dens_mass['Liq'])))

    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')


def print_reports(m):

    print('')
    for unit_k in [m.fs.boiler,
                   m.fs.reheater[1],
                   m.fs.reheater[2],
                   m.fs.bfp, m.fs.bfpt,
                   m.fs.booster,
                   m.fs.condenser_mix,
                   m.fs.charge.vhp_source_disjunct.ess_vhp_split,
                   m.fs.charge.solar_salt_disjunct.hxc,
                   m.fs.charge.hitec_salt_disjunct.hxc,
                   m.fs.charge.thermal_oil_disjunct.hxc,
                   m.fs.charge.connector,
                   m.fs.charge.cooler_connector,
                   m.fs.charge.cooler]:
        unit_k.display()

    for k in pyo.RangeSet(11):
        m.fs.turbine[k].report()
    for k in pyo.RangeSet(11):
        m.fs.turbine[k].display()
    for j in pyo.RangeSet(9):
        m.fs.fwh[j].report()
    for j in m.set_fwh_mixer:
        m.fs.fwh_mixer[j].display()


def model_analysis(m, solver, heat_duty=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    # Fix variables in the flowsheet
    m.fs.plant_power_out.fix(400)
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)

    m.fs.charge.solar_salt_disjunct.hxc.heat_duty.fix(heat_duty*1e6)  # in W
    m.fs.charge.hitec_salt_disjunct.hxc.heat_duty.fix(heat_duty*1e6)  # in W
    m.fs.charge.thermal_oil_disjunct.hxc.heat_duty.fix(heat_duty*1e6)  # in W

    # Unfix variables fixed in model input and during initialization
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s
    # m.fs.boiler.inlet.pressure.unfix()
    # m.fs.boiler.inlet.enth_mol.unfix()
    # m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s

    m.fs.charge.vhp_source_disjunct.ess_vhp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.charge.hp_source_disjunct.ess_hp_split.split_fraction[0, "to_hxc"].unfix()

    for salt_hxc in [m.fs.charge.solar_salt_disjunct.hxc,
                     m.fs.charge.hitec_salt_disjunct.hxc,
                     m.fs.charge.thermal_oil_disjunct.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    # Unfix outlet pressure of pump
    m.fs.charge.hx_pump.outlet.pressure[0].unfix()

    # Unfix variables fixed during initialization

    for unit in [m.fs.charge.connector,
                 m.fs.charge.cooler_disjunct.cooler,
                 m.fs.charge.cooler_connector,
                 m.fs.charge.hx_pump]:
        unit.inlet.unfix()
    m.fs.charge.cooler_disjunct.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    m.fs.production_cons.deactivate()
    # Add constraint to calculate net plant power output
    @m.fs.charge.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return m.fs.plant_power_out[t] == (
            (-1) * sum(m.fs.turbine[p].work_mechanical[t] for p in m.set_turbine) -
            b.hx_pump.control_volume.work[0]
        ) * (pyunits.MW / 1e6 * pyunits.W)

    # Disconnect arc included in the sink disjuncts
    for arc_s in [m.fs.fwh9_to_boiler]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()

    # Add scaling when scaling_factor is not used
    set_var_scaling(m)
    print('>> Set scaling factor through iscale')

    # Calculate revenue
    m.fs.lmp_signal = pyo.Param(initialize=22,
                                doc="Electricity price in $/MWh")
    m.fs.revenue = Expression(
        expr=(m.fs.lmp_signal * m.fs.plant_power_out[0]),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    # Objective function: total costs
    m.obj = Objective(
        expr=(
            m.fs.revenue -
            (m.fs.charge.operating_cost +
             m.fs.charge.plant_fixed_operating_cost +
             m.fs.charge.plant_variable_operating_cost) / (365 * 24) -
            (m.fs.charge.capital_cost +
             m.fs.charge.plant_capital_cost +
             m.fs.charge.cooler_capital_cost) / (365 * 24)
        ) * scaling_obj,
        sense=maximize
    )

    print('DOF before solution = ', degrees_of_freedom(m))

    # Solve the design optimization model
    # results = run_nlps(m,
    #                    solver=solver,
    #                    fluid="thermal_oil",
    #                    source="vhp",
    #                    sink="recycle_mix",
    #                    cooler=False)

    results = run_gdp(m)

    print_results(m, results)
    return m


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "tol": 1e-4,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    method = "with_efficiency"
    max_power = 436
    heat_duty_data = [150]
    for k in heat_duty_data:
        m_usc = usc.build_plant_model()
        usc.initialize(m_usc)
        
        m_chg, solver = main(m_usc, method=method, max_power=max_power)
        
        m = model_analysis(m_chg, solver, heat_duty=k)
