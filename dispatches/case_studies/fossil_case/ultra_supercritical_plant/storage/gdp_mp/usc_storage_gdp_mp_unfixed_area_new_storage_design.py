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
(3) Multi-stage turbines are modeled as multiple lumped single stage
    turbines

"""

__author__ = "Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
import json

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, Var, SolverFactory)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.network.plugins import expand_arcs
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
from pyomo.contrib.fbbt.fbbt import _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression

# Import IDAES libraries
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util import model_serializer as ms
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import (HeatExchanger,
                                      MomentumMixingType,
                                      Heater)
from idaes.models.unit_models import PressureChanger
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.unit_models.helm import (HelmMixer,
                                                                  HelmTurbineStage,
                                                                  HelmSplitter)
from idaes.core import UnitModelCostingBlock
from idaes.models.costing.SSLW import SSLWCosting, SSLWCostingData
from idaes.core.util.exceptions import ConfigurationError

# Import ultra supercritical power plant model
# from dispatches.models.fossil_case.ultra_supercritical_plant import (
#     ultra_supercritical_powerplant_mixcon as usc)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

# Import Solar salt property package
from dispatches.properties import solarsalt_properties

from IPython import embed
logging.basicConfig(level=logging.INFO)


# Open json file to add data to the model
with open('uscp_design_data_new_storage_design.json') as design_data:
    design_data_dict = json.load(design_data)


def create_gdp_model(m,
                     method=None,
                     max_power=None,
                     deact_arcs_after_init=None,
                     energy_loss=None):
    """Create flowsheet and add unit models.
    """

    # Add data to the model
    add_data(m)

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    ###########################################################################
    # Add global variables
    ###########################################################################

    m.fs.salt_amount = pyo.Param(
        initialize=m.max_salt_amount,
        doc="Solar salt amount in mton"
    )

    m.fs.salt_storage = pyo.Var(
        bounds=(-m.max_salt_flow, m.max_salt_flow),
        initialize=1,
        doc="Solar salt amount in storage"
    )
    m.fs.hx_pump_work = pyo.Var(
        bounds=(0, 1e3),
        initialize=1
    )
    m.fs.discharge_turbine_work = pyo.Var(
        bounds=(0, 1e3),
        initialize=1
    )

    if energy_loss:
        energy_loss_val = 1.5
    else:
        energy_loss_val = 0
    m.fs.energy_loss = pyo.Param(
        initialize=energy_loss_val,
        doc="Discharge energy loss in MW"
    )

    m.fs.charge_area = pyo.Var(
        initialize=m.hxc_area_init,
        bounds=(m.min_area, m.max_area),
        doc="Charge heat exchanger area in m2"
    )
    m.fs.hot_salt_temp = pyo.Var(
        initialize=m.hot_salt_temp,
        bounds=(m.min_salt_temp, m.max_salt_temp),
        doc="Hot salt temperature from charge heat exchanger in K"
    )
    m.fs.discharge_area = pyo.Var(
        initialize=m.hxd_area_init,
        bounds=(m.min_area, m.max_area),
        doc="Discharge heat exchanger area in m2"
    )
    ###########################################################################
    # Add disjunction
    ###########################################################################

    m.fs.discharge_mode_disjunct = Disjunct(rule=discharge_mode_disjunct_equations)
    m.fs.charge_mode_disjunct = Disjunct(rule=charge_mode_disjunct_equations)
    m.fs.no_storage_mode_disjunct = Disjunct(rule=no_storage_mode_disjunct_equations)

    ###########################################################################
    # Add constraints
    ###########################################################################

    if deact_arcs_after_init:
        print('           **Arcs from reheater 1 to turbine 3 and BFP to FWH8 are deactivated after initialization')
    else:
        print('           **Arcs from reheater 1 to turbine 3 and BFP to FWH8 are deactivated in create_gdp_model before initialization')
        _deactivate_arcs(m)

    _make_constraints(m, method=method, max_power=max_power)
    return m

def add_data(m):

    # Add global data
    m.hxc_area = design_data_dict["hxc_area"] # in MW
    m.hxd_area = design_data_dict["hxd_area"] # in MW
    m.min_power = design_data_dict["plant_min_power"] # in MW
    m.max_power = design_data_dict["plant_max_power"] # in MW
    m.ramp_rate = design_data_dict["ramp_rate"]
    m.min_power_storage = design_data_dict["min_discharge_turbine_power"] # in MW
    m.max_power_storage = design_data_dict["max_discharge_turbine_power"] # in MW
    m.hot_salt_temp = design_data_dict["hot_salt_temperature"] # in K
    # m.min_area = design_data_dict["min_storage_area_design"] # in m2
    m.min_area = 1000
    m.max_area = design_data_dict["max_storage_area_design"] # in m2
    m.cold_salt_temp = design_data_dict["cold_salt_temperature"] # in K
    m.min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    m.max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    m.min_salt_temp = design_data_dict["min_solar_salt_temperature"] # in K
    m.max_salt_temp = design_data_dict["max_solar_salt_temperature"] # in K
    m.max_salt_flow = design_data_dict["max_salt_flow"] # in kg/s
    m.factor_mton = design_data_dict["factor_mton"] # factor to convert kg to metric ton
    m.max_salt_amount = design_data_dict["max_salt_amount"] * m.factor_mton # in mton

    # Add initial values
    m.hxc_area_init = m.hxc_area
    m.hxd_area_init = m.hxd_area

    # Chemical engineering cost index for 2019
    m.CE_index = 607.5

    # Define the number of hours per day to operate the storage system
    # and the number of years over which the capital costs are
    # annualized
    m.fs.hours_per_day = pyo.Param(
        initialize=design_data_dict["operating_hours_per_day"],
        doc='Estimated number of hours of charging per day'
    )
    m.fs.num_of_years = pyo.Param(
        initialize=design_data_dict["number_of_years"],
        doc='Number of years for capital cost annualization')

    #  Define the data for the design of the storage heat
    # exchangers. The design is: Shell-n-tube counter-flow heat
    # exchanger design parameters. Data to compute overall heat
    # transfer coefficient for the charge heat exchanger using the
    # Sieder-Tate Correlation. Parameters for tube diameter and
    # thickness assumed from the data in (2017) He et al., Energy
    # Procedia 105, 980-985.
    m.fs.data_storage_hx = {
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    m.fs.tube_inner_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['tube_inner_dia'],
        units=pyunits.m,
        doc='Tube inner diameter in m')
    m.fs.tube_outer_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['tube_outer_dia'],
        units=pyunits.m,
        doc='Tube outer diameter in m')
    m.fs.k_steel = pyo.Param(
        initialize=m.fs.data_storage_hx['k_steel'],
        units=pyunits.W / (pyunits.m * pyunits.K),
        doc='Thermal conductivity of steel in W/m.K')
    m.fs.n_tubes = pyo.Param(
        initialize=m.fs.data_storage_hx['number_tubes'],
        doc='Number of tubes')
    m.fs.shell_inner_dia = pyo.Param(
        initialize=m.fs.data_storage_hx['shell_inner_dia'],
        units=pyunits.m,
        doc='Shell inner diameter in m')

    # Calculate sectional area of storage heat exchangers
    m.fs.tube_cs_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.tube_inner_dia ** 2),
        doc="Tube cross sectional area")
    m.fs.tube_out_area = pyo.Expression(
        expr=(pi / 4) *
        (m.fs.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness in m2")
    m.fs.shell_eff_area = pyo.Expression(
        expr=(
            (pi / 4) *
            (m.fs.shell_inner_dia ** 2) -
            m.fs.n_tubes *
            m.fs.tube_out_area),
        doc="Effective shell cross sectional area in m2")

    # Calculate ratios for overall heat transfer coefficients
    m.fs.tube_dia_ratio = (m.fs.tube_outer_dia / m.fs.tube_inner_dia)
    m.fs.log_tube_dia_ratio = log(m.fs.tube_dia_ratio)

    # Data for main flowsheet operation. The q baseline_charge
    # corresponds to heat duty of a plant with no storage and
    # producing 400 MW power
    m.data_cost = {
        'coal_price': 2.11e-9,
        'solar_salt_price': 0.49,
    }
    m.fs.coal_price = pyo.Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) in $/J')
    m.fs.solar_salt_price = pyo.Param(
        initialize=m.data_cost['solar_salt_price'],
        doc='Solar salt price in $/kg')


def _make_constraints(m, method=None, max_power=None):

    m.fs.production_cons.deactivate()
    @m.fs.Constraint(m.fs.time)
    def production_cons_with_storage(b, t):
        return (
            (
                (-1e-6) * (pyunits.MW / pyunits.W) *
                sum(b.turbine[p].work_mechanical[t] for p in m.set_turbine)
                - b.hx_pump_work
            ) ==
            b.plant_power_out[t]
        )

    m.fs.net_power = pyo.Expression(
        expr=(
            m.fs.plant_power_out[0]
            + m.fs.discharge_turbine_work
        )
    )

    m.fs.boiler_efficiency = pyo.Var(
        initialize=0.9,
        bounds=(0, 1),
        doc="Boiler efficiency in fraction"
    )
    m.fs.boiler_efficiency_eq = pyo.Constraint(
        expr=m.fs.boiler_efficiency == (
            0.2143 * (m.fs.net_power / max_power)
            + 0.7357
        ),
        doc="Boiler efficiency in fraction"
    )

    # m.fs.coal_heat_duty = pyo.Var(
    #     initialize=1000,
    #     bounds=(0, 1e5),
    #     doc="Coal heat duty supplied to boiler in MW")

    if method == "with_efficiency":
        # m.fs.coal_heat_duty_eq = pyo.Constraint(
        #     expr=m.fs.coal_heat_duty * m.fs.boiler_efficiency ==
        #     m.fs.plant_heat_duty[0]
        # )
        m.fs.coal_heat_duty = pyo.Expression(
            expr=m.fs.plant_heat_duty[0] / m.fs.boiler_efficiency
        )
    else:
        # m.fs.coal_heat_duty_eq = pyo.Constraint(
        #     expr=m.fs.coal_heat_duty ==
        #     m.fs.plant_heat_duty[0]
        # )
        m.fs.coal_heat_duty = pyo.Expression(
            expr=m.fs.plant_heat_duty[0]
        )

    m.fs.cycle_efficiency = pyo.Var(
        initialize=0.4,
        bounds=(0, 1),
        doc="Cycle efficiency in fraction"
    )
    m.fs.cycle_efficiency_eq = pyo.Constraint(
        expr=m.fs.cycle_efficiency * m.fs.coal_heat_duty == m.fs.net_power,
        doc="Cycle efficiency in fraction"
    )


def add_disjunction(m):
    """Add storage fluid selection and steam source disjunctions to the
    model
    """

    m.fs.operation_mode_disjunction = Disjunction(
        expr=[m.fs.no_storage_mode_disjunct,
              m.fs.charge_mode_disjunct,
              m.fs.discharge_mode_disjunct])

    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs)

    return m


def no_storage_mode_disjunct_equations(disj):
    m = disj.model()

    # Connect cycle
    m.fs.no_storage_mode_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from reheater 1 to turbine 3"
    )
    m.fs.no_storage_mode_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet,
        doc="Connection from FWH9 to boiler"
    )
    m.fs.no_storage_mode_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].tube_inlet,
        doc="Connection from condenser pump to FWH1"
    )

    # Set global constraints that depend on the charge and discharge
    # operation modes to zero since the units do not exist in no
    # storage mode
    m.fs.no_storage_mode_disjunct.eq_salt_amount_in_storage = pyo.Constraint(
        expr=m.fs.salt_storage == 0
    )
    # m.fs.no_storage_mode_disjunct.eq_cooler_heat_duty = pyo.Constraint(
    #     expr=m.fs.cooler_heat_duty == 0
    # )
    m.fs.no_storage_mode_disjunct.eq_hx_pump_work = pyo.Constraint(
        expr=m.fs.hx_pump_work == 0
    )
    m.fs.no_storage_mode_disjunct.eq_discharge_turbine_work = pyo.Constraint(
        expr=m.fs.discharge_turbine_work == 0
    )


def charge_mode_disjunct_equations(disj):
    m = disj.model()

    # Declare units for the charge storage system: A splitter to
    # divert some steam from high pressure inlet and intermediate
    # pressure inlet to charge the storage heat exchanger, a charge
    # heat exchanger, a pump, and a mixer. A pump is used to increase
    # the pressure of the water to allow mixing it at a desired
    # location within the plant.
    m.fs.charge_mode_disjunct.ess_charge_split = HelmSplitter(
        property_package=m.fs.prop_water,
        outlet_list=["to_hxc", "to_turbine"]
    )

    m.fs.charge_mode_disjunct.hxc = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.prop_water},
        tube={"property_package": m.fs.solar_salt_properties}
    )

    m.fs.charge_mode_disjunct.hx_pump = PressureChanger(
        property_package=m.fs.prop_water,
        material_balance_type=MaterialBalanceType.componentTotal,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    m.fs.charge_mode_disjunct.mixer2 = HelmMixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["from_fwh9", "from_hx_pump"],
        property_package=m.fs.prop_water,
    )

    # Calculate the overall heat transfer coefficient for the Solar
    # salt charge heat exchanger. For that, first calculate Reynolds
    # number, Prandtl number, and Nusselt number.
    solar_hxc = m.fs.charge_mode_disjunct.hxc
    solar_hxc.salt_reynolds_number = pyo.Expression(
        expr=(
            (solar_hxc.tube_inlet.flow_mass[0] *
             m.fs.tube_outer_dia) /
            (m.fs.shell_eff_area *
             solar_hxc.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number")
    solar_hxc.salt_prandtl_number = pyo.Expression(
        expr=(
            solar_hxc.cold_side.properties_in[0].cp_mass["Liq"] *
            solar_hxc.cold_side.properties_in[0].visc_d_phase["Liq"] /
            solar_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number")
    solar_hxc.salt_prandtl_wall = pyo.Expression(
        expr=(
            solar_hxc.cold_side.properties_out[0].cp_mass["Liq"] *
            solar_hxc.cold_side.properties_out[0].visc_d_phase["Liq"] /
            solar_hxc.cold_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number at wall")
    solar_hxc.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 *
            (solar_hxc.salt_reynolds_number**0.6) *
            (solar_hxc.salt_prandtl_number**0.4) *
            ((solar_hxc.salt_prandtl_number /
              solar_hxc.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")
    solar_hxc.steam_reynolds_number = pyo.Expression(
        expr=(
            solar_hxc.shell_inlet.flow_mol[0] *
            solar_hxc.hot_side.properties_in[0].mw *
            m.fs.tube_inner_dia /
            (m.fs.tube_cs_area *
             m.fs.n_tubes *
             solar_hxc.hot_side.properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")
    solar_hxc.steam_prandtl_number = pyo.Expression(
        expr=(
            (solar_hxc.hot_side.properties_in[0].cp_mol /
             solar_hxc.hot_side.properties_in[0].mw) *
            solar_hxc.hot_side.properties_in[0].visc_d_phase["Vap"] /
            solar_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")
    solar_hxc.steam_nusselt_number = pyo.Expression(
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
    solar_hxc.h_salt = pyo.Expression(
        expr=(
            solar_hxc.cold_side.properties_in[0].therm_cond_phase["Liq"] *
            solar_hxc.salt_nusselt_number /
            m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    solar_hxc.h_steam = pyo.Expression(
        expr=(
            solar_hxc.hot_side.properties_in[0].therm_cond_phase["Vap"] *
            solar_hxc.steam_nusselt_number /
            m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    @m.fs.charge_mode_disjunct.hxc.Constraint(
        doc="Solar salt charge heat exchanger overall heat transfer coefficient")
    def constraint_hxc_ohtc(b):
        return (
            b.overall_heat_transfer_coefficient[0] *
            (2 * m.fs.k_steel *
             b.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             b.h_salt *
             b.h_steam +
             m.fs.tube_dia_ratio *
             b.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              b.h_salt *
              b.h_steam)


    # Add constraint to ensure that the mixer 2 outlet is at the
    # minimum pressure
    m.fs.charge_mode_disjunct.mixer2_pressure_constraint = pyo.Constraint(
        expr=m.fs.charge_mode_disjunct.mixer2.from_fwh9_state[0].pressure ==
        m.fs.charge_mode_disjunct.mixer2.mixed_state[0].pressure,
        doc="Mixer 2 outlet pressure equal to min inlet pressure")

    # Add pump pressure constraint
    m.fs.charge_mode_disjunct.constraint_hxpump_presout = pyo.Constraint(
        expr=m.fs.charge_mode_disjunct.hx_pump.outlet.pressure[0] >=
        m.main_steam_pressure * 1.1231
        # expr=m.fs.charge_mode_disjunct.hx_pump.outlet.pressure[0] ==
        # m.main_steam_pressure * 1.1231
    )

    # Reconnect condenser pump to FWH1
    m.fs.charge_mode_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].tube_inlet,
        doc="Connection from condenser pump to FWH1"
    )

    # Declare arcs to connect storage charge system to the plant
    m.fs.charge_mode_disjunct.rh1_to_esscharg = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.charge_mode_disjunct.ess_charge_split.inlet,
        doc="Connection from reheater 1 to HP splitter"
    )
    m.fs.charge_mode_disjunct.esscharg_to_turb3 = Arc(
        source=m.fs.charge_mode_disjunct.ess_charge_split.to_turbine,
        destination=m.fs.turbine[3].inlet,
        doc="Connection from HP splitter to turbine 3"
    )
    m.fs.charge_mode_disjunct.esscharg_to_hxc = Arc(
        source=m.fs.charge_mode_disjunct.ess_charge_split.to_hxc,
        destination=m.fs.charge_mode_disjunct.hxc.shell_inlet,
        doc="Connection from HP splitter to HXC inlet 1"
    )
    m.fs.charge_mode_disjunct.hxc_to_hxpump = Arc(
        source=m.fs.charge_mode_disjunct.hxc.shell_outlet,
        destination=m.fs.charge_mode_disjunct.hx_pump.inlet,
        doc="Connection from HXC to HX pump"
    )

    # Declare arcs to connect the mixer 2 to the plant
    m.fs.charge_mode_disjunct.fwh9_to_mix2 = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.charge_mode_disjunct.mixer2.from_fwh9,
        doc="Connection from FWH9 outlet to mixer2"
    )
    m.fs.charge_mode_disjunct.hxpump_to_mix2 = Arc(
        source=m.fs.charge_mode_disjunct.hx_pump.outlet,
        destination=m.fs.charge_mode_disjunct.mixer2.from_hx_pump,
        doc="Connection from HX pump to mixer 2"
    )
    m.fs.charge_mode_disjunct.mix2_to_boiler = Arc(
        source=m.fs.charge_mode_disjunct.mixer2.outlet,
        destination=m.fs.boiler.inlet,
        doc="Connection from fwh9 outlet to boiler"
    )

    # Declare constraints to save global variables
    m.fs.charge_mode_disjunct.eq_salt_amount_in_charge_storage = pyo.Constraint(
        expr=m.fs.salt_storage == m.fs.charge_mode_disjunct.hxc.tube_inlet.flow_mass[0]
    )
    m.fs.charge_mode_disjunct.eq_hx_pump_work = pyo.Constraint(
        expr=m.fs.hx_pump_work == (
            (1e-6) * (pyunits.MW / pyunits.W) *
            m.fs.charge_mode_disjunct.hx_pump.control_volume.work[0]
        )
    )
    m.fs.charge_mode_disjunct.eq_discharge_turbine_work = pyo.Constraint(
        expr=m.fs.discharge_turbine_work == 0
    )

    # m.fs.charge_mode_disjunct.eq_charge_heat_duty = pyo.Constraint(
    #     expr=(
    #         (1e-6) * (pyunits.MW / pyunits.W) *
    #         m.fs.charge_mode_disjunct.hxc.heat_duty[0]
    #     ) <= m.max_storage_heat_duty
    # )

    # Save area and hot salt temperature in global variable
    m.fs.charge_mode_disjunct.eq_charge_area = pyo.Constraint(
        expr=m.fs.charge_area == m.fs.charge_mode_disjunct.hxc.area
    )
    m.fs.charge_mode_disjunct.eq_hot_salt_temperature = pyo.Constraint(
        expr=m.fs.hot_salt_temp == m.fs.charge_mode_disjunct.hxc.tube_outlet.temperature[0]
    )


def discharge_mode_disjunct_equations(disj):
    m = disj.model()

    # Declare units for the discharge storage system: A splitter to
    # divert some condensate from the feed water heater train to be
    # heated up in the discharge heat exchanger, a discharge heat
    # exchanger, and a turbine to produce extra energy.
    m.fs.discharge_mode_disjunct.ess_discharge_split = HelmSplitter(
        property_package=m.fs.prop_water,
        outlet_list=["to_hxd", "to_fwh1"]
    )

    m.fs.discharge_mode_disjunct.hxd = HeatExchanger(
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={"property_package": m.fs.solar_salt_properties},
        tube={"property_package": m.fs.prop_water}
    )

    m.fs.discharge_mode_disjunct.es_turbine = HelmTurbineStage(
        property_package=m.fs.prop_water
    )
    m.fs.discharge_mode_disjunct.eq_turbine_temperature_out = pyo.Constraint(
        expr=(
            m.fs.discharge_mode_disjunct.es_turbine.control_volume.properties_out[0].temperature ==
            m.fs.discharge_mode_disjunct.es_turbine.control_volume.properties_out[0].temperature_sat + 1
        )
    )

    # Calculate the overall heat transfer coefficient for the Solar
    # salt charge heat exchanger. For that, first calculate Reynolds
    # number, Prandtl number, and Nusselt number.
    solar_hxd = m.fs.discharge_mode_disjunct.hxd
    solar_hxd.salt_reynolds_number = pyo.Expression(
        expr=(
            solar_hxd.shell_inlet.flow_mass[0]
            * m.fs.tube_outer_dia
            / (m.fs.shell_eff_area
               * solar_hxd.hot_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Salt Reynolds Number"
    )
    solar_hxd.salt_prandtl_number = pyo.Expression(
        expr=(
            solar_hxd.hot_side.properties_in[0].cp_mass["Liq"]
            * solar_hxd.hot_side.properties_in[0].visc_d_phase["Liq"]
            / solar_hxd.hot_side.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Salt Prandtl Number"
    )
    # Assuming that the wall conditions are same as those at the outlet
    solar_hxd.salt_prandtl_wall = pyo.Expression(
        expr=(
            solar_hxd.hot_side.properties_out[0].cp_mass["Liq"]
            * solar_hxd.hot_side.properties_out[0].visc_d_phase["Liq"]
            / solar_hxd.hot_side.properties_out[0].therm_cond_phase["Liq"]
        ),
        doc="Wall Salt Prandtl Number"
    )
    solar_hxd.salt_nusselt_number = pyo.Expression(
        expr=(
            0.35 * (solar_hxd.salt_reynolds_number**0.6)
            * (solar_hxd.salt_prandtl_number**0.4)
            * ((solar_hxd.salt_prandtl_number
                / solar_hxd.salt_prandtl_wall)**0.25)
            * (2**0.2)
        ),
        doc="Solar Salt Nusslet Number from 2019, App Ener (233-234), 126"
    )
    solar_hxd.steam_reynolds_number = pyo.Expression(
        expr=(
            solar_hxd.tube_inlet.flow_mol[0]
            * solar_hxd.cold_side.properties_in[0].mw
            * m.fs.tube_inner_dia
            / (m.fs.tube_cs_area
               * m.fs.n_tubes
               * solar_hxd.cold_side.properties_in[0].visc_d_phase["Liq"])
        ),
        doc="Steam Reynolds Number"
    )
    solar_hxd.steam_prandtl_number = pyo.Expression(
        expr=(
            (solar_hxd.cold_side.properties_in[0].cp_mol
             / solar_hxd.cold_side.properties_in[0].mw)
            * solar_hxd.cold_side.properties_in[0].visc_d_phase["Liq"]
            / solar_hxd.cold_side.properties_in[0].therm_cond_phase["Liq"]
        ),
        doc="Steam Prandtl Number"
    )
    solar_hxd.steam_nusselt_number = pyo.Expression(
        expr=(
            0.023 * (solar_hxd.steam_reynolds_number ** 0.8)
            * (solar_hxd.steam_prandtl_number ** (0.33))
            * ((solar_hxd.cold_side.properties_in[0].visc_d_phase["Liq"]
                / solar_hxd.cold_side.properties_out[0].visc_d_phase["Vap"]
                ) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia"
    )

    # Calculate discharge heat exchanger salt and steam side heat
    # transfer coefficients
    solar_hxd.h_salt = pyo.Expression(
        expr=(
            solar_hxd.hot_side.properties_in[0].therm_cond_phase["Liq"]
            * solar_hxd.salt_nusselt_number / m.fs.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]"
    )
    solar_hxd.h_steam = pyo.Expression(
        expr=(
            solar_hxd.cold_side.properties_in[0].therm_cond_phase["Liq"]
            * solar_hxd.steam_nusselt_number / m.fs.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]"
    )

    @m.fs.discharge_mode_disjunct.hxd.Constraint(
        doc="Solar salt discharge heat exchanger overall heat transfer coefficient")
    def constraint_hxd_ohtc(b):
        return (
            b.overall_heat_transfer_coefficient[0] *
            (2 * m.fs.k_steel *
             b.h_steam +
             m.fs.tube_outer_dia *
             m.fs.log_tube_dia_ratio *
             b.h_salt *
             b.h_steam +
             m.fs.tube_dia_ratio *
             b.h_salt *
             2 * m.fs.k_steel)
        ) == (2 * m.fs.k_steel *
              b.h_salt *
              b.h_steam)

    # Reconnect arcs that were disconnected in the global model
    m.fs.discharge_mode_disjunct.rh1_to_turb3 = Arc(
        source=m.fs.reheater[1].outlet,
        destination=m.fs.turbine[3].inlet
    )
    m.fs.discharge_mode_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].tube_outlet,
        destination=m.fs.boiler.inlet
    )

    # Declare arcs to connect discharge heat exchanger to plant
    m.fs.discharge_mode_disjunct.condpump_to_essdisch = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.discharge_mode_disjunct.ess_discharge_split.inlet,
        doc="Connection from condenser pump outlet to BFP splitter"
    )
    m.fs.discharge_mode_disjunct.essdisch_to_fwh1 = Arc(
        source=m.fs.discharge_mode_disjunct.ess_discharge_split.to_fwh1,
        destination=m.fs.fwh[1].tube_inlet,
        doc="Connection from condenser pump splitter to FWH1"
    )
    m.fs.discharge_mode_disjunct.essdisch_to_hxd = Arc(
        source=m.fs.discharge_mode_disjunct.ess_discharge_split.to_hxd,
        destination=m.fs.discharge_mode_disjunct.hxd.tube_inlet,
        doc="Connection from condenser pump splitter to discharge heat exchanger"
    )
    m.fs.discharge_mode_disjunct.hxd_to_esturbine = Arc(
        source=m.fs.discharge_mode_disjunct.hxd.tube_outlet,
        destination=m.fs.discharge_mode_disjunct.es_turbine.inlet,
        doc="Connection from discharge heat exchanger to ES turbine"
    )


    # Save the amount of salt used in the discharge heat exchanger
    m.fs.discharge_mode_disjunct.eq_salt_amount_in_discharge_storage = pyo.Constraint(
        expr=m.fs.salt_storage == -m.fs.discharge_mode_disjunct.hxd.shell_inlet.flow_mass[0]
    )

    # Fix HX pump work to zero since it does not exist during
    # discharge mode and the value is saved in a global variable
    m.fs.discharge_mode_disjunct.eq_hx_pump_work = pyo.Constraint(
        expr=m.fs.hx_pump_work == 0
    )

    m.fs.discharge_mode_disjunct.eq_discharge_turbine_work = pyo.Constraint(
        expr=m.fs.discharge_turbine_work == (
            (-1e-6) * (pyunits.MW / pyunits.W) *
            m.fs.discharge_mode_disjunct.es_turbine.work[0]
        )
    )

    # m.fs.discharge_mode_disjunct.eq_discharge_heat_duty = pyo.Constraint(
    #     expr=(
    #         (1e-6) * (pyunits.MW / pyunits.W) *
    #         m.fs.discharge_mode_disjunct.hxd.heat_duty[0] +
    #         m.fs.energy_loss
    #     ) <= m.max_storage_heat_duty
    # )

    m.fs.discharge_mode_disjunct.eq_charge_area = pyo.Constraint(
        expr=m.fs.discharge_area == m.fs.discharge_mode_disjunct.hxd.area
    )


def _deactivate_arcs(m):
    """Deactivate arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger
    for arc_s in [m.fs.rh1_to_turb3,
                  m.fs.fwh9_to_boiler,
                  m.fs.condpump_to_fwh1]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def set_model_input(m):
    """Define model inputs and fixed variables or parameter values

    The parameter values in this block, unless otherwise stated
    explicitly, are either assumed or estimated for a total power out
    of 437 MW.

    Unless stated otherwise, the units are: temperature in K, pressure
    in Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    in W

    """

    ###########################################################################
    #  Storage heat exchanger section
    ###########################################################################
    # Add heat exchanger area from supercritical plant model_input. For
    # conceptual design optimization, area is unfixed and optimized
    m.fs.charge_mode_disjunct.hxc.area.fix(2500)
    m.fs.discharge_mode_disjunct.hxd.area.fix(2500)

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.charge_mode_disjunct.hxc.tube_inlet.flow_mass.fix(140)
    m.fs.charge_mode_disjunct.hxc.tube_inlet.temperature.fix(513.15)
    m.fs.charge_mode_disjunct.hxc.tube_inlet.pressure.fix(101325)

    m.fs.discharge_mode_disjunct.hxd.shell_inlet.flow_mass.fix(200)
    m.fs.discharge_mode_disjunct.hxd.shell_inlet.temperature.fix(853.15)
    m.fs.discharge_mode_disjunct.hxd.shell_inlet.pressure.fix(101325)

    # HX pump efficiecncy assumption
    m.fs.charge_mode_disjunct.hx_pump.efficiency_pump.fix(0.80)
    # m.fs.charge.hx_pump.outlet.pressure[0].fix(m.main_steam_pressure * 1.1231)

    # m.fs.discharge_mode_disjunct.es_turbine.ratioP.fix(0.0286)
    m.fs.discharge_mode_disjunct.es_turbine.efficiency_isentropic.fix(0.8)

    ###########################################################################
    #  ESS VHP and HP splitters                                               #
    ###########################################################################
    # The model is built for a fixed flow of steam through the
    # charger.  This flow of steam to the charger is unfixed and
    # determine during design optimization
    m.fs.charge_mode_disjunct.ess_charge_split.split_fraction[0, "to_hxc"].fix(0.1)
    m.fs.discharge_mode_disjunct.ess_discharge_split.split_fraction[0, "to_hxd"].fix(0.1)

    # Fix global variables
    # m.fs.hx_pump_work.fix(0)
    # m.fs.discharge_turbine_work.fix(0)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for Solar salt charge and discharge heat
    # exchanger.
    for fluid in [m.fs.charge_mode_disjunct.hxc,
                  m.fs.discharge_mode_disjunct.hxd]:
        iscale.set_scaling_factor(fluid.area, 1e-2)
        iscale.set_scaling_factor(
            fluid.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(fluid.shell.heat, 1e-6)
        iscale.set_scaling_factor(fluid.tube.heat, 1e-6)

    iscale.set_scaling_factor(m.fs.charge_mode_disjunct.hx_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.discharge_mode_disjunct.es_turbine.control_volume.work, 1e-6)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def set_var_scaling(m):
    iscale.set_scaling_factor(m.fs.fuel_cost, 1e-3)
    iscale.set_scaling_factor(m.fs.plant_fixed_operating_cost, 1e-3)
    iscale.set_scaling_factor(m.fs.plant_variable_operating_cost, 1e-3)
    # iscale.set_scaling_factor(m.fs.plant_capital_cost, 1e-3)

    iscale.set_scaling_factor(m.fs.salt_amount, 1e-3)
    iscale.set_scaling_factor(m.fs.salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(m.fs.salt_inventory_cold, 1e-3)
    iscale.set_scaling_factor(m.fs.previous_salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(m.fs.previous_salt_inventory_cold, 1e-3)

    iscale.set_scaling_factor(m.fs.constraint_salt_inventory_hot, 1e-3)

    iscale.set_scaling_factor(m.fs.charge_mode_disjunct.capital_cost, 1e-3)
    iscale.set_scaling_factor(m.fs.discharge_mode_disjunct.capital_cost, 1e-3)
    # iscale.set_scaling_factor(m.fs.storage_capital_cost, 1e-3)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)


def initialize(m,
               solver=None,
               deact_arcs_after_init=None,
               outlvl=idaeslog.WARNING,
               optarg={"tol": 1e-8, "max_iter": 300}):
    """Initialize the units included in the charge model
    """
    print()
    print('>> Start initialization of charge units in ultra-supercritical plant')
    # print('   {} DOFs before initialization'.format(degrees_of_freedom(m)))

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    # Include scaling factors
    set_scaling_factors(m)
    # iscale.calculate_scaling_factors(m)

    # Initialize all units in charge mode operation
    propagate_state(m.fs.charge_mode_disjunct.rh1_to_esscharg)
    m.fs.charge_mode_disjunct.ess_charge_split.initialize(outlvl=outlvl,
                                                          optarg=solver.options)
    propagate_state(m.fs.charge_mode_disjunct.esscharg_to_hxc)
    m.fs.charge_mode_disjunct.hxc.initialize(outlvl=outlvl,
                                             optarg=solver.options)

    if not deact_arcs_after_init:
        # Reinitialize and fix turbine 3 inlet since the arc is
        # disconnected
        propagate_state(m.fs.charge_mode_disjunct.esscharg_to_turb3)
        m.fs.turbine[3].inlet.fix()
        m.fs.turbine[3].initialize(outlvl=outlvl,
                                   optarg=solver.options)

    propagate_state(m.fs.charge_mode_disjunct.hxc_to_hxpump)
    m.fs.charge_mode_disjunct.hx_pump.initialize(outlvl=outlvl,
                                                 optarg=solver.options)

    # Fix value of global variable
    m.fs.hx_pump_work.fix((1e-6) * (pyunits.MW / pyunits.W) *
                          m.fs.charge_mode_disjunct.hx_pump.control_volume.work[0].value)

    propagate_state(m.fs.charge_mode_disjunct.fwh9_to_mix2)
    propagate_state(m.fs.charge_mode_disjunct.hxpump_to_mix2)
    m.fs.charge_mode_disjunct.mixer2.initialize(outlvl=outlvl)

    # Initialize all units in discharge mode operation
    propagate_state(m.fs.discharge_mode_disjunct.condpump_to_essdisch)
    m.fs.discharge_mode_disjunct.ess_discharge_split.initialize(outlvl=outlvl,
                                                                optarg=solver.options)
    propagate_state(m.fs.discharge_mode_disjunct.essdisch_to_hxd)
    m.fs.discharge_mode_disjunct.hxd.initialize(outlvl=outlvl,
                                                optarg=solver.options)
    propagate_state(m.fs.discharge_mode_disjunct.hxd_to_esturbine)
    m.fs.discharge_mode_disjunct.es_turbine.initialize(outlvl=outlvl,
                                                       optarg=solver.options)
    # Fix value of global variable
    m.fs.discharge_turbine_work.fix((-1e-6) * (pyunits.MW / pyunits.W) *
                                    m.fs.discharge_mode_disjunct.es_turbine.work[0].value)

    if not deact_arcs_after_init:
        # Reinitialize FWH8 using bfp outlet
        m.fs.fwh[8].fwh_vfrac_constraint.deactivate()
        m.fs.fwh[8].tube_inlet.flow_mol.fix(m.fs.bfp.outlet.flow_mol[0])
        m.fs.fwh[8].tube_inlet.enth_mol.fix(m.fs.bfp.outlet.enth_mol[0])
        m.fs.fwh[8].tube_inlet.pressure.fix(m.fs.bfp.outlet.pressure[0])
        m.fs.fwh[8].initialize(outlvl=outlvl,
                               optarg=solver.options)
        m.fs.fwh[8].fwh_vfrac_constraint.activate()

    # Check and raise an error if the degrees of freedom are not 0
    # print('   {} DOFs before initialization solution'.format(degrees_of_freedom(m)))
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
            )

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)

    print("   **Solver termination for Charge Model Initialization:",
          res.solver.termination_condition)
    # print('   {} DOFs after initialization solution'.format(degrees_of_freedom(m)))
    print('>> End initialization of charge units in ultra-supercritical plant')
    print()


def build_costing(m):
    """Add cost correlations for the storage design analysis.

    This function is used to estimate the capital and operatig cost of
    integrating an energy storage system. It contains cost
    correlations to estimate: (i) the capital cost of charge heat
    exchanger and salt inventory and (ii) the operating costs for 1
    year.

    """

    ##############################################################
    # Add capital cost
    # 1. Calculate change and discharge heat exchangers costs
    # 2. Calculate total capital cost for charge and discharge heat
    # exchangers
    ##############################################################

    # Add IDAES costing method
    m.fs.costing = SSLWCosting()

    ###### 1. Calculate charge and discharge heat exchangers costs
    # Calculate and initialize the Solar salt charge and discharge
    # heat exchangers coss, which are estimated using the IDAES
    # costing method with default options, i.e., a U-tube heat
    # exchanger, stainless steel material, and a tube length of
    # 12ft. Refer to costing documentation to change any of the
    # default options. The purchase cost of heat exchanger has to be
    # annualized when used
    for storage_hx in [m.fs.charge_mode_disjunct.hxc,
                       m.fs.discharge_mode_disjunct.hxd]:
        storage_hx.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method=SSLWCostingData.cost_heat_exchanger
        )

    ###### 2. Calculate total capital cost for charge and discharge
    ###### heat exchangers
    m.fs.charge_mode_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Capital cost of charge heat exchanger in $/h")
    def charge_solar_cap_cost_rule(b):
        return b.capital_cost == (
            m.fs.charge_mode_disjunct.hxc.costing.capital_cost /
            (m.fs.num_of_years * 365 * 24)
        )
    m.fs.charge_mode_disjunct.cap_cost_eq = pyo.Constraint(
        rule=charge_solar_cap_cost_rule)

    m.fs.discharge_mode_disjunct.capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Capital cost of discharge heat exhcnager in $/h")
    def discharge_solar_cap_cost_rule(b):
        return b.capital_cost == (
            m.fs.discharge_mode_disjunct.hxd.costing.capital_cost /
            (m.fs.num_of_years * 365 * 24)
        )
    m.fs.discharge_mode_disjunct.cap_cost_eq = pyo.Constraint(
        rule=discharge_solar_cap_cost_rule)

    # Save total storage annual capital cost at global level
    m.fs.storage_capital_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e7),
        doc="Annualized capital cost for solar salt in $/h")

    m.fs.no_storage_mode_disjunct.capital_cost_eq_constraint = pyo.Constraint(
        expr=m.fs.storage_capital_cost == 0
    )
    m.fs.charge_mode_disjunct.capital_cost_eq_constraint = pyo.Constraint(
        expr=m.fs.storage_capital_cost == m.fs.charge_mode_disjunct.capital_cost
    )
    m.fs.discharge_mode_disjunct.capital_cost_eq_constraint = pyo.Constraint(
        expr=m.fs.storage_capital_cost == m.fs.discharge_mode_disjunct.capital_cost
    )

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.fuel_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Operating cost in $/h")

    def fuel_cost_rule(b):
        return b.fuel_cost == (
            b.operating_hours *
            b.coal_price *
            (b.coal_heat_duty * 1e6)
        ) / (365 * 24)
    m.fs.fuel_cost_eq = pyo.Constraint(rule=fuel_cost_rule)

    ###########################################################################
    #  Add capital and operating cost for full plant
    ###########################################################################

    # Calculate capital cost for power plant
    # m.fs.plant_capital_cost = pyo.Var(
    #     initialize=1000000,
    #     bounds=(0, 1e12),
    #     doc="Plant capital cost in $/hour")
    m.fs.plant_fixed_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant fixed operating cost in $/hour")
    m.fs.plant_variable_operating_cost = pyo.Var(
        initialize=1000000,
        bounds=(0, 1e12),
        doc="Plant variable operating cost in $/hour")

    # def plant_cap_cost_rule(b):
    #     return m.fs.plant_capital_cost == (
    #         ((2688973 * m.fs.plant_power_out[0]
    #           + 618968072) /
    #          (m.fs.num_of_years * 365 * 24)
    #         ) * (m.CE_index / 575.4)
    #     )
    # m.fs.plant_cap_cost_eq = pyo.Constraint(rule=plant_cap_cost_rule)

    def op_fixed_plant_cost_rule(b):
        return b.plant_fixed_operating_cost == (
            ((16657.5 * b.plant_power_out[0]
              + 6109833.3) /
             (b.num_of_years * 365 * 24)
            ) * (m.CE_index / 575.4)
        )
    m.fs.op_fixed_plant_cost_eq = pyo.Constraint(rule=op_fixed_plant_cost_rule)

    def op_variable_plant_cost_rule(b):
        return b.plant_variable_operating_cost == (
            (31754.7 * b.plant_power_out[0]
            ) * (m.CE_index / 575.4)
        ) / (365 * 24)
    m.fs.op_variable_plant_cost_eq = pyo.Constraint(
        rule=op_variable_plant_cost_rule)

    return m


def initialize_with_costing(m):

    optarg = {
        "tol": 1e-8,
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Fix operating cost variable to initialize cost in a square
    # problem
    # m.fs.fuel_cost.fix(1e6)

    # Initialize capital costs for charge and discharge heat
    # exchangers
    calculate_variable_from_constraint(
        m.fs.charge_mode_disjunct.capital_cost,
        m.fs.charge_mode_disjunct.cap_cost_eq)
    calculate_variable_from_constraint(
        m.fs.discharge_mode_disjunct.capital_cost,
        m.fs.discharge_mode_disjunct.cap_cost_eq)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.fuel_cost,
        m.fs.fuel_cost_eq)

    # # Initialize capital cost of power plant
    # calculate_variable_from_constraint(
    #     m.fs.plant_capital_cost,
    #     m.fs.plant_cap_cost_eq)

    # Initialize plant fixed and variable operating costs
    calculate_variable_from_constraint(
        m.fs.plant_fixed_operating_cost,
        m.fs.op_fixed_plant_cost_eq)
    calculate_variable_from_constraint(
        m.fs.plant_variable_operating_cost,
        m.fs.op_variable_plant_cost_eq)

    print()
    print('>> Start initialization of costing correlations')

    # Check and raise an error if the degrees of freedom are not 0
    # print('   {} DOFs before cost initialization'.format(degrees_of_freedom(m)))
    if not degrees_of_freedom(m) == 0:
        raise ConfigurationError(
            "The degrees of freedom after building the model are not 0. "
            "You have {} degrees of freedom. "
            "Please check your inputs to ensure a square problem "
            "before initializing the model.".format(degrees_of_freedom(m))
            )

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)
    print("   **Solver termination in cost initialization: ",
          res.solver.termination_condition)
    print('>> End initialization of costing correlations')
    print()


def calculate_bounds(m):
    m.fs.temperature_degrees = 5

    # Calculate bounds for solar salt from properties expressions
    m.fs.solar_salt_temperature_max = 853.15 + m.fs.temperature_degrees # in K
    m.fs.solar_salt_temperature_min = 513.15 - m.fs.temperature_degrees # in K
    # Note: min/max interchanged because at max temperature we obtain the min value
    m.fs.solar_salt_enth_mass_max = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_max - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.solar_salt_temperature_max - 273.15)**2)
    )
    m.fs.solar_salt_enth_mass_min = (
        (m.fs.solar_salt_properties.cp_param_1.value *
         (m.fs.solar_salt_temperature_min - 273.15))
        + (m.fs.solar_salt_properties.cp_param_2.value * 0.5 * \
           (m.fs.solar_salt_temperature_min - 273.15)**2)
    )

    m.fs.salt_enth_mass_max = m.fs.solar_salt_enth_mass_max
    m.fs.salt_enth_mass_min = m.fs.solar_salt_enth_mass_min

    # print('   **Calculate bounds for solar salt')
    # print('     Mass enthalpy max: {: >4.4f}, min: {: >4.4f}'.format(
    #     m.fs.solar_salt_enth_mass_max, m.fs.solar_salt_enth_mass_min))


def add_bounds(m):
    """Add bounds to units in charge model

    """

    calculate_bounds(m)

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W
    m.flow_max = m.main_flow * 3       # Units in mol/s
    m.flow_min = 11804                   # Units in mol/s
    m.heat_duty_max = (m.max_storage_heat_duty * 1e6)  # Units in MW
    m.factor = 2.5
    m.flow_max_storage = 0.2 * m.flow_max
    m.flow_min_storage = 1e-3

    # Turbines
    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)

    # Booster
    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_k.inlet.flow_mol[:].setub(m.flow_max)
        unit_k.outlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_k.outlet.flow_mol[:].setub(m.flow_max)

    # Turbine splitters flow
    for k in m.set_turbine_splitter:
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setlb(m.flow_min_storage)
        m.fs.turbine_splitter[k].inlet.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setlb(m.flow_min_storage)
        m.fs.turbine_splitter[k].outlet_1.flow_mol[:].setub(m.flow_max)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setlb(m.flow_min_storage)
        m.fs.turbine_splitter[k].outlet_2.flow_mol[:].setub(m.flow_max)

    # Add bounds to all units in charge mode
    for unit_in_charge in [m.fs.charge_mode_disjunct]:
        # Charge heat exchanger (HXC)
        unit_in_charge.hxc.shell_inlet.flow_mol.setlb(m.flow_min_storage)
        unit_in_charge.hxc.shell_inlet.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.hxc.tube_inlet.flow_mass.setlb(m.flow_min_storage)
        unit_in_charge.hxc.tube_inlet.flow_mass.setub(m.max_salt_flow)
        unit_in_charge.hxc.shell_outlet.flow_mol.setlb(m.flow_min_storage)
        unit_in_charge.hxc.shell_outlet.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.hxc.tube_outlet.flow_mass.setlb(m.flow_min_storage)
        unit_in_charge.hxc.tube_outlet.flow_mass.setub(m.max_salt_flow)
        unit_in_charge.hxc.tube_inlet.pressure.setlb(101320)
        unit_in_charge.hxc.tube_inlet.pressure.setub(101330)
        unit_in_charge.hxc.tube_outlet.pressure.setlb(101320)
        unit_in_charge.hxc.tube_outlet.pressure.setub(101330)
        unit_in_charge.hxc.heat_duty.setlb(0)
        unit_in_charge.hxc.heat_duty.setub(m.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setlb(-m.heat_duty_max)
        unit_in_charge.hxc.shell.heat.setub(0)
        unit_in_charge.hxc.tube.heat.setlb(0)
        unit_in_charge.hxc.tube.heat.setub(m.heat_duty_max)
        unit_in_charge.hxc.tube.properties_in[:].enth_mass.setlb(
            m.fs.salt_enth_mass_min / m.factor)
        unit_in_charge.hxc.tube.properties_in[:].enth_mass.setub(
            m.fs.salt_enth_mass_max * m.factor)
        unit_in_charge.hxc.tube.properties_out[:].enth_mass.setlb(
            m.fs.salt_enth_mass_min / m.factor)
        unit_in_charge.hxc.tube.properties_out[:].enth_mass.setub(
            m.fs.salt_enth_mass_max * m.factor)
        # unit_in_charge.hxc.tube.properties_in[:].enth_mass.setlb(1)
        # unit_in_charge.hxc.tube.properties_in[:].enth_mass.setub(1.5e6)
        # unit_in_charge.hxc.tube.properties_out[:].enth_mass.setlb(1)
        # unit_in_charge.hxc.tube.properties_out[:].enth_mass.setub(1.5e6)
        unit_in_charge.hxc.overall_heat_transfer_coefficient.setlb(1)
        unit_in_charge.hxc.overall_heat_transfer_coefficient.setub(10000)
        unit_in_charge.hxc.area.setlb(m.min_area)
        unit_in_charge.hxc.area.setub(m.max_area)
        unit_in_charge.hxc.delta_temperature_in.setlb(9)
        unit_in_charge.hxc.delta_temperature_out.setlb(5)
        unit_in_charge.hxc.delta_temperature_in.setub(120)
        unit_in_charge.hxc.delta_temperature_out.setub(100)
        unit_in_charge.hxc.costing.pressure_factor.setlb(0)
        unit_in_charge.hxc.costing.pressure_factor.setub(1e6)
        unit_in_charge.hxc.costing.capital_cost.setlb(0)
        unit_in_charge.hxc.costing.capital_cost.setub(1e8)
        unit_in_charge.hxc.costing.base_cost_per_unit.setlb(0)
        unit_in_charge.hxc.costing.base_cost_per_unit.setub(1e8)
        unit_in_charge.hxc.costing.material_factor.setlb(0)
        unit_in_charge.hxc.costing.material_factor.setub(100)

        # HX pump
        for unit_k in [unit_in_charge.hx_pump]:
            unit_k.inlet.flow_mol.setlb(0)
            unit_k.inlet.flow_mol.setub(m.flow_max_storage)
            unit_k.outlet.flow_mol.setlb(0)
            unit_k.outlet.flow_mol.setub(m.flow_max_storage)
            unit_k.deltaP.setlb(0)
            unit_k.deltaP.setub(1e10)
        unit_in_charge.hx_pump.work_mechanical[0].setlb(0)
        unit_in_charge.hx_pump.work_mechanical[0].setub(1e10)
        unit_in_charge.hx_pump.ratioP.setlb(0)
        unit_in_charge.hx_pump.ratioP.setub(100)
        unit_in_charge.hx_pump.work_fluid[0].setlb(0)
        unit_in_charge.hx_pump.work_fluid[0].setub(1e8)
        unit_in_charge.hx_pump.efficiency_pump[0].setlb(0)
        unit_in_charge.hx_pump.efficiency_pump[0].setub(1)

        # HP splitter
        unit_in_charge.ess_charge_split.inlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_charge.ess_charge_split.inlet.flow_mol[:].setub(m.flow_max)
        unit_in_charge.ess_charge_split.to_hxc.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_charge.ess_charge_split.to_hxc.flow_mol[:].setub(m.flow_max_storage)
        unit_in_charge.ess_charge_split.to_turbine.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_charge.ess_charge_split.to_turbine.flow_mol[:].setub(m.flow_max)
        unit_in_charge.ess_charge_split.split_fraction[0.0, "to_hxc"].setlb(0)
        unit_in_charge.ess_charge_split.split_fraction[0.0, "to_hxc"].setub(1)
        unit_in_charge.ess_charge_split.split_fraction[0.0, "to_turbine"].setlb(0)
        unit_in_charge.ess_charge_split.split_fraction[0.0, "to_turbine"].setub(1)

        # Mixer 2
        unit_in_charge.mixer2.from_fwh9.flow_mol.setlb(m.flow_min_storage)
        unit_in_charge.mixer2.from_fwh9.flow_mol.setub(m.flow_max)
        unit_in_charge.mixer2.from_hx_pump.flow_mol.setlb(0)
        unit_in_charge.mixer2.from_hx_pump.flow_mol.setub(m.flow_max_storage)
        unit_in_charge.mixer2.outlet.flow_mol.setlb(0)
        unit_in_charge.mixer2.outlet.flow_mol.setub(m.flow_max)


    # Add bounds to all units in discharge mode
    for unit_in_discharge in [m.fs.discharge_mode_disjunct]:
        # Discharge heat exchanger (HXD)
        unit_in_discharge.hxd.shell_inlet.flow_mass.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.shell_inlet.flow_mass.setub(m.max_salt_flow)
        unit_in_discharge.hxd.shell_outlet.flow_mass.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.shell_outlet.flow_mass.setub(m.max_salt_flow)
        unit_in_discharge.hxd.tube_inlet.flow_mol.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.tube_inlet.flow_mol.setub(m.flow_max_storage)
        unit_in_discharge.hxd.tube_outlet.flow_mol.setlb(m.flow_min_storage)
        unit_in_discharge.hxd.tube_outlet.flow_mol.setub(m.flow_max_storage)
        unit_in_discharge.hxd.shell_inlet.pressure.setlb(101320)
        unit_in_discharge.hxd.shell_inlet.pressure.setub(101330)
        unit_in_discharge.hxd.shell_outlet.pressure.setlb(101320)
        unit_in_discharge.hxd.shell_outlet.pressure.setub(101330)
        unit_in_discharge.hxd.heat_duty.setlb(0)
        unit_in_discharge.hxd.heat_duty.setub(m.heat_duty_max)
        unit_in_discharge.hxd.tube.heat.setlb(0)
        unit_in_discharge.hxd.tube.heat.setub(m.heat_duty_max)
        unit_in_discharge.hxd.shell.heat.setlb(-m.heat_duty_max)
        unit_in_discharge.hxd.shell.heat.setub(0)
        unit_in_discharge.hxd.shell.properties_in[:].enth_mass.setlb(
            m.fs.salt_enth_mass_min / m.factor)
        unit_in_discharge.hxd.shell.properties_in[:].enth_mass.setub(
            m.fs.salt_enth_mass_max * m.factor)
        unit_in_discharge.hxd.shell.properties_out[:].enth_mass.setlb(
            m.fs.salt_enth_mass_min / m.factor)
        unit_in_discharge.hxd.shell.properties_out[:].enth_mass.setub(
            m.fs.salt_enth_mass_max * m.factor)
        # unit_in_discharge.hxd.shell.properties_in[:].enth_mass.setlb(1)
        # unit_in_discharge.hxd.shell.properties_in[:].enth_mass.setub(1.5e6)
        # unit_in_discharge.hxd.shell.properties_out[:].enth_mass.setlb(1)
        # unit_in_discharge.hxd.shell.properties_out[:].enth_mass.setub(1.5e6)
        unit_in_discharge.hxd.overall_heat_transfer_coefficient.setlb(1)
        unit_in_discharge.hxd.overall_heat_transfer_coefficient.setub(10000)
        unit_in_discharge.hxd.area.setlb(m.min_area)
        unit_in_discharge.hxd.area.setub(m.max_area)
        unit_in_discharge.hxd.delta_temperature_in.setlb(5)
        unit_in_discharge.hxd.delta_temperature_out.setlb(10)
        unit_in_discharge.hxd.delta_temperature_in.setub(350)
        unit_in_discharge.hxd.delta_temperature_out.setub(500)
        unit_in_discharge.hxd.costing.pressure_factor.setlb(0)
        unit_in_discharge.hxd.costing.pressure_factor.setub(1e5)
        unit_in_discharge.hxd.costing.capital_cost.setlb(0)
        unit_in_discharge.hxd.costing.capital_cost.setub(1e8)
        unit_in_discharge.hxd.costing.base_cost_per_unit.setlb(0)
        unit_in_discharge.hxd.costing.base_cost_per_unit.setub(1e8)
        unit_in_discharge.hxd.costing.material_factor.setlb(0)
        unit_in_discharge.hxd.costing.material_factor.setub(100)


        # BFP splitter
        unit_in_discharge.ess_discharge_split.inlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_discharge.ess_discharge_split.inlet.flow_mol[:].setub(m.flow_max)
        unit_in_discharge.ess_discharge_split.to_hxd.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_discharge.ess_discharge_split.to_hxd.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.ess_discharge_split.to_fwh1.flow_mol[:].setlb(0)
        unit_in_discharge.ess_discharge_split.to_fwh1.flow_mol[:].setub(m.flow_max)
        unit_in_discharge.ess_discharge_split.split_fraction[0.0, "to_hxd"].setlb(0)
        unit_in_discharge.ess_discharge_split.split_fraction[0.0, "to_hxd"].setub(1)
        unit_in_discharge.ess_discharge_split.split_fraction[0.0, "to_fwh1"].setlb(0)
        unit_in_discharge.ess_discharge_split.split_fraction[0.0, "to_fwh1"].setub(1)

        # ES Turbine
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_discharge.es_turbine.inlet.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setlb(m.flow_min_storage)
        unit_in_discharge.es_turbine.outlet.flow_mol[:].setub(m.flow_max_storage)
        unit_in_discharge.es_turbine.deltaP.setlb(-1e10)
        unit_in_discharge.es_turbine.deltaP.setub(1e10)
        unit_in_discharge.es_turbine.work.setlb(-1e12)
        unit_in_discharge.es_turbine.work.setub(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setlb(0)
        unit_in_discharge.es_turbine.efficiency_isentropic.setub(1)
        unit_in_discharge.es_turbine.ratioP.setlb(0)
        unit_in_discharge.es_turbine.ratioP.setub(100)
        unit_in_discharge.es_turbine.efficiency_mech.setlb(0)
        unit_in_discharge.es_turbine.efficiency_mech.setub(1)
        unit_in_discharge.es_turbine.shaft_speed.setlb(0)
        unit_in_discharge.es_turbine.shaft_speed.setub(1000)


def main(method=None,
         max_power=None,
         load_init_file=None,
         path_init_file=None,
         deact_arcs_after_init=None,
         energy_loss=None,
         solver=None):

    if load_init_file:
        # Build ultra-supercritical plant model and initialize it
        m = usc.build_plant_model()

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_gdp_model(m,
                             method=method,
                             max_power=max_power,
                             deact_arcs_after_init=deact_arcs_after_init,
                             energy_loss=energy_loss)

        # Set required inputs to the model to have a square problem for
        # initialization
        set_model_input(m)

        # Add scaling factors
        set_scaling_factors(m)

        # Add cost correlations
        m = build_costing(m)

        # Initialize using .json file (with bounds)
        ms.from_json(m, fname=path_init_file)
        print()
        print('>>>>> Initializing model using .json file: {}'.format(path_init_file))

        # Add bounds
        add_bounds(m)

    else:
        # Build ultra-supercritical plant model and initialize it
        m = usc.build_plant_model()
        print()
        print('>> Start initialization of ultra-supercritical plant base model')
        usc.initialize(m)
        print('>> End initialization of ultra-supercritical plant base model')

        # Create a flowsheet, add properties, unit models, and arcs
        m = create_gdp_model(m,
                             method=method,
                             max_power=max_power,
                             deact_arcs_after_init=deact_arcs_after_init,
                             energy_loss=energy_loss)

        # Set required inputs to the model to have a square problem for
        # initialization
        set_model_input(m)

        # Add scaling factors
        set_scaling_factors(m)

        # Initialize the model with a sequential initialization and custom
        # routines
        initialize(m, deact_arcs_after_init=deact_arcs_after_init)

        # Add cost correlations
        m = build_costing(m)
        # print('DOF after costing: ', degrees_of_freedom(m))

        # Initialize costing
        initialize_with_costing(m)

        # Add bounds
        add_bounds(m)

        # Calculate and store initialization file
        ms.to_json(m, fname=path_init_file)
        print()
        print('>>>>> Saving initialization .json file in {}'.format(path_init_file))


    # Add disjunctions
    add_disjunction(m)

    if deact_arcs_after_init:
        # Deactivate arcs
        _deactivate_arcs(m)

    return m


def print_results(m, results):

    m.fs.condenser_mix.makeup.display()

    print('================================')
    print()
    print("***************** Optimization Results ******************")
    print('Obj ($/h): {:.4f}'.format(value(m.obj) / m.scaling_obj))
    print('Revenue ($/h): {:.4f}'.format(
        value(m.fs.revenue)))
    # print('Plant capital cost ($/h): {:.4f}'.format(
    #     value(m.fs.plant_capital_cost)))
    print('Plant fixed operating costs ($/h): {:.4f}'.format(
        value(m.fs.plant_fixed_operating_cost)))
    print('Plant variable operating costs ($/h): {:.4f}'.format(
        value(m.fs.plant_variable_operating_cost)))
    print('Coal Cost (fuel) ($/h): {:.4f}'.format(
        value(m.fs.fuel_cost)))
    print('Storage Capital Cost ($/h): {:.4f}'.format(
        value(m.fs.storage_capital_cost)))
    print()
    print("***************** Tank Results ******************")
    print('Hot Salt Inventory (mton): {:.4f}, prev: {:.4f}'.format(
        value(m.fs.salt_inventory_hot),
        value(m.fs.previous_salt_inventory_hot)))
    print('Cold Salt Inventory (mton): {:.4f}, prev: {:.4f}'.format(
        value(m.fs.salt_inventory_cold),
        value(m.fs.previous_salt_inventory_cold)))
    print('Salt to storage (mton): {:.4f}'.format(
        value(m.fs.salt_storage)))
    print('Salt Amount (mton): {:.4f}'.format(
        value(m.fs.salt_amount)))
    print('')
    print("***************** Power Plant Operation ******************")
    print('')
    print('Net Power (MW): {:.4f}'.format(
        value(m.fs.net_power)))
    print('Plant Power (MW): {:.4f}'.format(
        value(m.fs.plant_power_out[0])))
    print('Discharge turbine power (MW) [ES turbine Power]: {:.4f} [{:.4f}]'.format(
        value(m.fs.discharge_turbine_work),
        value(m.fs.discharge_mode_disjunct.es_turbine.work_mechanical[0]) * (-1e-6)))
    print('HX pump work (MW): {:.4f}'.format(
        value(m.fs.hx_pump_work)))
    print('Boiler feed water flow (mol/s): {:.4f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler (plant) heat duty (MW_th): {:.4f}'.format(
        value(m.fs.plant_heat_duty[0])))
    print('Makeup water flow: {:.4f}'.format(
        value(m.fs.condenser_mix.makeup.flow_mol[0])))
    print()
    # print('Boiler efficiency (%): boiler: {:.4f}'.format(
    #     value(m.fs.boiler_efficiency) * 100))
    print('Boiler/Cycle efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
        value(m.fs.boiler_efficiency) * 100,
        value(m.fs.cycle_efficiency) * 100))
    print()
    if m.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
        print("***************** Charge Heat Exchanger (HXC) ******************")
        print('HXC area (m2): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.area)))
        print('HXC heat duty (MW): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6))
        print('HXC salt flow (kg/s): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.tube_inlet.flow_mass[0])))
        print('HXC steam flow to storage (mol/s): {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.shell_inlet.flow_mol[0])))
        print('HXC salt temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.tube_inlet.temperature[0]),
            value(m.fs.charge_mode_disjunct.hxc.tube_outlet.temperature[0])))
        print('HXC water temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.hot_side.properties_in[0].temperature),
            value(m.fs.charge_mode_disjunct.hxc.hot_side.properties_out[0].temperature)))
        print('HXC delta temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_in[0]),
            value(m.fs.charge_mode_disjunct.hxc.delta_temperature_out[0])))
        print('')
    elif m.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
        print("*************** Discharge Heat Exchanger (HXD) ****************")
        print('')
        print('HXD area (m2): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.area)))
        print('HXD heat duty (MW): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6))
        print('HXD salt flow (kg/s): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.shell_inlet.flow_mass[0])))
        print('HXD Steam flow to storage (mol/s): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.tube_inlet.flow_mol[0])))
        print('HXD salt temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.shell_inlet.temperature[0]),
            value(m.fs.discharge_mode_disjunct.hxd.shell_outlet.temperature[0])))
        print('HXD water temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.cold_side.properties_in[0].temperature),
            value(m.fs.discharge_mode_disjunct.hxd.cold_side.properties_out[0].temperature)))
        print('HXD delta temperature (K): in: {:.4f}, out: {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_in[0]),
            value(m.fs.discharge_mode_disjunct.hxd.delta_temperature_out[0])))
        print('ES Turbine work (MW): {:.4f}'.format(
            value(m.fs.discharge_mode_disjunct.es_turbine.work[0]) * -1e-6))
        print('')

    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')


def print_reports(m):

    print('')
    for unit_k in [m.fs.boiler, m.fs.reheater[1],
                   m.fs.reheater[2],
                   m.fs.bfp, m.fs.bfpt,
                   m.fs.booster,
                   m.fs.condenser_mix,
                   m.fs.charge.hxc]:
        unit_k.display()

    for k in RangeSet(11):
        m.fs.turbine[k].report()
    for k in RangeSet(11):
        m.fs.turbine[k].display()
    for j in RangeSet(9):
        m.fs.fwh[j].report()
    for j in m.set_fwh_mixer:
        m.fs.fwh_mixer[j].display()

def print_model(_, nlp_model, nlp_data):

    print('       ___________________________________________')
    if nlp_model.fs.charge_mode_disjunct.indicator_var.value == 1:
        print('        Disjunction 1: Charge mode is selected')
        print('         HXC area (m2): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.area)))
        print('         HXC heat duty (MW): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6))
        print('         HXC salt flow (kg/s): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.tube_inlet.flow_mass[0])))
        print('         HXC steam flow (mol/s): {:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.shell_inlet.flow_mol[0])))
        print('         HXC Salt temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.tube_inlet.temperature[0]),
            value(nlp_model.fs.charge_mode_disjunct.hxc.tube_outlet.temperature[0])))
        print('         HXC Delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(nlp_model.fs.charge_mode_disjunct.hxc.delta_temperature_in[0]),
            value(nlp_model.fs.charge_mode_disjunct.hxc.delta_temperature_out[0])))
    elif nlp_model.fs.discharge_mode_disjunct.indicator_var.value == 1:
        print('        Disjunction 1: Discharge mode is selected')
        print('         HXD area (m2): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.area)))
        print('         HXD heat duty (MW): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6))
        print('         HXD salt flow (kg/s): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.shell_inlet.flow_mass[0])))
        print('         HXD steam flow (mol/s): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.tube_inlet.flow_mol[0])))
        print('         HXD Salt temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.shell_inlet.temperature[0]),
            value(nlp_model.fs.discharge_mode_disjunct.hxd.shell_outlet.temperature[0])))
        print('         HXD Delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.hxd.delta_temperature_in[0]),
            value(nlp_model.fs.discharge_mode_disjunct.hxd.delta_temperature_out[0])))
        print('         ES turbine work (MW): {:.4f}'.format(
            value(nlp_model.fs.discharge_mode_disjunct.es_turbine.work_mechanical[0]) * (-1e-6)))
    elif nlp_model.fs.no_storage_mode_disjunct.indicator_var.value == 1:
        print('        Disjunction 1: No storage mode is selected')
    else:
        print('        No other operation alternative!')

    print()
    print('        Obj (M$/year): {:.4f}'.format(
        value(nlp_model.obj) / nlp_model.scaling_obj))
    print('        Revenue ($/h): {:.4f}'.format(
        value(nlp_model.fs.revenue)))
    # print('        Plant capital cost ($/h): {:.4f}'.format(
    #     value(nlp_model.fs.plant_capital_cost)))
    print('        Plant fixed operating costs ($/h): {:.4f}'.format(
        value(nlp_model.fs.plant_fixed_operating_cost)))
    print('        Plant variable operating costs ($/h): {:.4f}'.format(
        value(nlp_model.fs.plant_variable_operating_cost)))
    print('        Coal Cost (fuel) ($/h): {:.4f}'.format(
        value(nlp_model.fs.fuel_cost)))
    print('        Storage Capital Cost ($/h): {:.4f}'.format(
        value(nlp_model.fs.storage_capital_cost)))
    print('        Net Power (MW): {:.4f}'.format(
        value(nlp_model.fs.net_power)))
    print('        Plant Power (MW): {:.4f}'.format(
        value(nlp_model.fs.plant_power_out[0])))
    print('        Discharge turbine work (MW): {:.4f}'.format(
        value(nlp_model.fs.discharge_turbine_work)))
    print('        HX pump work (MW): {:.4f}'.format(
        value(nlp_model.fs.hx_pump_work)))
    # print('        Boiler efficiency (%): {:.4f}'.format(
    #     value(nlp_model.fs.boiler_efficiency) * 100))
    print('        Boiler/cycle efficiency (%): {:.4f}/{:.4f}'.format(
        value(nlp_model.fs.boiler_efficiency) * 100,
        value(nlp_model.fs.cycle_efficiency) * 100))
    print('        Hot Previous Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.previous_salt_inventory_hot)))
    print('        Cold Previous Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.previous_salt_inventory_cold)))
    print('        Salt to storage (kg/s) [mton]: {:.4f} [{:.4f}]'.format(
        value(nlp_model.fs.salt_storage),
        value(nlp_model.fs.salt_storage) * 3600 * nlp_model.factor_mton))
    print('        Hot Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.salt_inventory_hot)))
    print('        Cold Salt Inventory (mton): {:.4f}'.format(
        value(nlp_model.fs.salt_inventory_cold)))


    print('       ___________________________________________')

    log_close_to_bounds(nlp_model)
    # log_infeasible_constraints(nlp_model)


def run_nlps(m,
             solver=None,
             operation_mode=None):
    """This function fixes the indicator variables of the disjuncts so to
    solve NLP problems

    """

    print()
    print('>>> You are solving an NLP problem by fixing the operation disjuncts!')
    if operation_mode == "charge":
        print('           ** Solving for charge mode')
        m.fs.charge_mode_disjunct.indicator_var.fix(True)
        m.fs.discharge_mode_disjunct.indicator_var.fix(False)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(False)
    elif operation_mode == "discharge":
        print('           ** Solving for discharge mode')
        m.fs.charge_mode_disjunct.indicator_var.fix(False)
        m.fs.discharge_mode_disjunct.indicator_var.fix(True)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(False)
    elif operation_mode == "no_storage":
        print('           ** Solving for no storage mode')
        m.fs.charge_mode_disjunct.indicator_var.fix(False)
        m.fs.discharge_mode_disjunct.indicator_var.fix(False)
        m.fs.no_storage_mode_disjunct.indicator_var.fix(True)
    else:
        print('<(x.x)> Unrecognized operation mode! Try charge, discharge, or no_storage')
    print()
    print()

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

    log_close_to_bounds(m)
    log_infeasible_constraints(m)

    print_results(m, results)

    return m, results


def run_gdp(m):

    print('>>> You are solving GDP model using GDPopt')
    print('    {} DOFs before solving GDP model '.format(degrees_of_freedom(m)))

    opt = SolverFactory('gdpopt')
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None

    results = opt.solve(
        m,
        tee=True,
        algorithm='RIC',
        # OA_penalty_factor=1e4,
        # max_slack=1e4,
        call_after_subproblem_solve=print_model,
        mip_solver='gurobi_direct',
        nlp_solver='ipopt',
        init_algorithm="no_init",
        time_limit="2400",
        subproblem_presolve=False,
        nlp_solver_args=dict(
            tee=True,
            symbolic_solver_labels=True,
            options={
                "linear_solver": "ma27",
                "max_iter": 150
            }
        )
    )

    print_results(m, results)
    # print_reports(m)

    return results


def model_analysis(m,
                   solver=None,
                   power=None,
                   max_power=None,
                   tank_scenario=None,
                   fix_power=None,
                   operation_mode=None,
                   method=None,
                   deact_arcs_after_init=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    if fix_power:
        m.fs.power_demand_eq = pyo.Constraint(
            expr=m.fs.net_power == power
        )
    else:
        m.fs.plant_power_min = pyo.Constraint(
            expr=m.fs.plant_power_out[0] >= m.min_power
        )
        m.fs.plant_power_max = pyo.Constraint(
            expr=m.fs.plant_power_out[0] <= max_power
        )
        hxc_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * m.fs.charge_mode_disjunct.hxc.heat_duty[0]
        hxd_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * m.fs.discharge_mode_disjunct.hxd.heat_duty[0]
        m.fs.charge_mode_disjunct.storage_min_heat_duty = pyo.Constraint(
            expr=hxc_heat_duty >= m.min_storage_heat_duty
        )
        m.fs.discharge_mode_disjunct.storage_min_heat_duty = pyo.Constraint(
            expr=hxd_heat_duty >= m.min_storage_heat_duty
        )
        # m.fs.charge_mode_disjunct.storage_max_heat_duty = pyo.Constraint(
        #     expr=hxc_heat_duty <= m.max_storage_heat_duty
        # )
        # m.fs.discharge_mode_disjunct.storage_max_heat_duty = pyo.Constraint(
        #     expr=hxd_heat_duty <= m.max_storage_heat_duty * (1 - 0.01)
        # )

    # Fix and unfix boiler data
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.boiler.inlet.flow_mol.unfix()

    # Unfix data fixed during initialization
    # m.fs.fuel_cost.unfix()
    m.fs.charge_mode_disjunct.ess_charge_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.discharge_mode_disjunct.ess_discharge_split.split_fraction[0, "to_hxd"].unfix()

    if not deact_arcs_after_init:
        m.fs.turbine[3].inlet.unfix()
        m.fs.fwh[8].tube_inlet.unfix()

    for salt_hxc in [m.fs.charge_mode_disjunct.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()
        salt_hxc.area.unfix()

    for salt_hxd in [m.fs.discharge_mode_disjunct.hxd]:
        salt_hxd.tube_inlet.unfix()
        salt_hxd.shell_inlet.flow_mass.unfix()
        salt_hxd.area.unfix()

    # Unfix global variables
    m.fs.hx_pump_work.unfix()
    m.fs.discharge_turbine_work.unfix()

    # Fix storage heat exchangers design
    m.fs.charge_mode_disjunct.hxc.tube_outlet.temperature[0].fix(m.hot_salt_temp)
    m.fs.discharge_mode_disjunct.hxd.shell_inlet.temperature[0].fix(m.hot_salt_temp)
    m.fs.discharge_mode_disjunct.hxd.shell_outlet.temperature[0].fix(m.cold_salt_temp)

    # Add salt inventory variables
    min_tank = 1 * m.factor_mton # in mton
    max_tank = m.max_salt_amount - min_tank # in mton
    max_inventory = 1e7 * m.factor_mton # in mton
    min_inventory = 75000 * m.factor_mton # in mton

    # Add variables and mass balances for the hot storage tank
    m.fs.previous_salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=min_inventory,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the beginning of time period in mton"
        )
    m.fs.salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=min_inventory,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the end of time period in mton"
        )
    m.fs.previous_salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=max_tank - min_inventory,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the beginning of time period in mton"
        )
    m.fs.salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=max_tank - min_inventory,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the end of time period in mton"
        )

    @m.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return b.salt_inventory_hot == (
            b.previous_salt_inventory_hot
            + (3600 * m.fs.salt_storage) * m.factor_mton
        )

    @m.fs.Constraint(doc="Maximum previous salt inventory at any time")
    def constraint_salt_inventory(b):
        return b.salt_amount == (
            b.salt_inventory_hot
            + b.salt_inventory_cold
        )

    # Fix the previous salt inventory based on the tank scenario
    if tank_scenario == "hot_empty":
        m.fs.previous_salt_inventory_hot.fix(min_tank)
        m.fs.previous_salt_inventory_cold.fix(max_tank)
    elif tank_scenario == "hot_half_full":
        m.fs.previous_salt_inventory_hot.fix(max_tank / 2)
        m.fs.previous_salt_inventory_cold.fix(max_tank / 2)
    elif tank_scenario == "hot_full":
        m.fs.previous_salt_inventory_hot.fix(max_tank)
        m.fs.previous_salt_inventory_cold.fix(min_tank)
    else:
        print('Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full')

    # Add LMP data
    m.fs.lmp = pyo.Var(
        m.fs.time,
        domain=Reals,
        initialize=80,
        doc="Hourly LMP in $/MWh"
        )

    # Fix LMP data according to the case we want to solve. When
    # solving GDP model, a random value is selected
    if operation_mode == "charge":
        m_chg.fs.lmp[0].fix(22.9684)
    elif operation_mode == "discharge":
        m_chg.fs.lmp[0].fix(200)
    elif operation_mode == "no_storage":
        m_chg.fs.lmp[0].fix(50)
    else:
        m_chg.fs.lmp[0].fix(22.9684)
        print('   **Use fixed LMP signal value of {} $/MWh'.format(
            value(m.fs.lmp[0])))
        print()


    m.fs.revenue = pyo.Expression(
        expr=(m.fs.lmp[0] * m.fs.net_power),
        doc="Revenue function in $/h assuming 1 hr operation"
    )

    # Set scaling factors to variables including during model analysis
    set_var_scaling(m)

    # Add a total cost function as the objective function. Also,
    # include a scaling factor to the objective.
    m.scaling_obj = 1e-3
    m.obj = Objective(
        expr=(
            m.fs.revenue -
            (m.fs.fuel_cost +
             m.fs.plant_fixed_operating_cost +
             m.fs.plant_variable_operating_cost) -
            (m.fs.storage_capital_cost
             # + m.fs.plant_capital_cost
            )
        ) * m.scaling_obj,
        sense=maximize
    )

    if operation_mode is not None:
        # Solve NLP problem with fix operation mode disjunct
        run_nlps(m,
                 solver=solver,
                 operation_mode=operation_mode)
    else:
        # Solve using GDPopt
        run_gdp(m)


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # How to run this model:
    #   load_init_file: Set to True if you wish to initialize using a .json file and
    #                   indicate the path of the .json file in path_init_file
    #   fix_power:      Select True if you want to fix the power output of the plant.
    #                   If True, then provide the power value in power_demand
    #   method:         Select between "with_efficiency" or "without_efficiency"
    #   tank_scenario:  Select the initial value for the salt tank levels:
    #                   "hot_empty", "hot_full", "hot_half_full" (hot refers to hot salt)
    #   operation_mode: Select None if you want to solve the GDP formulation (GDPopt solver)
    #                   If you wish to solve for one mode, select an operation mode and the
    #                   respective NLP problem is solved. The modes are: "charge", "discharge",
    #                   or "no_storage"
    #  deact_arcs_after_init: Set to True if you wish to deactivate the arcs that are
    #                         connecting reheater 1 to turbine 3 and bfp to FWH8 after
    #                         initialization. If False, the arcs are deactivated in
    #                         create_gdp_model and turbine 3 and FWH8 inlets are fixed during
    #                         initialization.

    max_power = design_data_dict["plant_max_power"] # in MW
    power_demand = 400 # in MW
    load_init_file = False
    if load_init_file:
        path_init_file = design_data_dict["gdp_init_file_path"]
    else:
        path_init_file = None

    fix_power = False
    method = "with_efficiency"
    tank_scenario = "hot_empty"
    operation_mode = None
    deact_arcs_after_init = True # when False, cost initialization takes about 20 sec more
    energy_loss = True

    m_chg = main(method=method,
                 max_power=max_power,
                 load_init_file=load_init_file,
                 path_init_file=path_init_file,
                 deact_arcs_after_init=deact_arcs_after_init,
                 energy_loss=energy_loss)

    m = model_analysis(m_chg,
                       solver,
                       power=power_demand,
                       max_power=max_power,
                       tank_scenario=tank_scenario,
                       fix_power=fix_power,
                       operation_mode=operation_mode,
                       method=method,
                       deact_arcs_after_init=deact_arcs_after_init)
