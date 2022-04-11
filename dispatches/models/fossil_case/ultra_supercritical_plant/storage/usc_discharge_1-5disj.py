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
    and condenser.  (3) Multi-stage turbines are modeled as multiple
    lumped single stage turbines

updated (08/12/2021)
"""

__author__ = "Naresh Susarla and Soraya Rawlings"

# Import Python libraries
from math import pi
import logging
# Import Pyomo libraries
import os
from pyomo.environ import (Block, Param, Constraint, Objective,
                           TransformationFactory, SolverFactory,
                           Expression, value, log, exp, Var)
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.gdp import Disjunct, Disjunction

# Import IDAES libraries
from idaes.core import MaterialBalanceType
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (HeatExchanger,
                                              MomentumMixingType,
                                              Heater)
import idaes.core.util.unit_costing as icost

# Import IDAES Libraries
from idaes.generic_models.unit_models import (
    Mixer,
    PressureChanger
)
# from idaes.generic_models.unit_models.separator import (Separator,
#                                                         SplittingType)
from idaes.power_generation.unit_models.helm import (
    HelmMixer,
    HelmTurbineStage,
    HelmSplitter
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback, HeatExchangerFlowPattern)
from idaes.generic_models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.misc import svg_tag

# Import ultra supercritical power plant model
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
import solarsalt_properties

from pyomo.network.plugins import expand_arcs

from IPython import embed

logging.basicConfig(level=logging.INFO)


def create_discharge_model(m):
    """Create flowsheet and add unit models.
    """

    # Create a block to add charge storage model
    m.fs.discharge = Block()

    # Add molten salt properties (Solar and Hitec salt)
    m.fs.solar_salt_properties = solarsalt_properties.SolarsaltParameterBlock()

    ###########################################################################
    #  Add vhp and hp splitters                                               #
    ###########################################################################
    # Declared to divert some steam from high pressure inlet and
    # intermediate pressure inlet to charge the storage heat exchanger
    m.fs.discharge.es_split = HelmSplitter(
        default={
            "property_package": m.fs.prop_water,
            "outlet_list": ["to_fwh", "to_hxd"]
        }
    )
    m.fs.discharge.hxd = HeatExchanger(
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.solar_salt_properties
            },
            "tube": {
                "property_package": m.fs.prop_water
            }
        }
    )

    m.fs.discharge.es_turbine = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    ###########################################################################
    #  Declare disjuncts
    ###########################################################################
    # Disjunction 1 for the sink of discharge HX consists of 2 disjuncts:
    #   1. ccs_sink_disjunct ======> steam from hxd is used in ccs
    #   2. plant_sink_disjunct ======> steam from hxd is used in the turbines

    m.fs.discharge.condpump_source_disjunct = Disjunct(
        rule=condpump_source_disjunct_equations)
    m.fs.discharge.fwh4_source_disjunct = Disjunct(
        rule=fwh4_source_disjunct_equations)
    m.fs.discharge.booster_source_disjunct = Disjunct(
        rule=booster_source_disjunct_equations)
    m.fs.discharge.bfp_source_disjunct = Disjunct(
        rule=bfp_source_disjunct_equations)
    m.fs.discharge.fwh9_source_disjunct = Disjunct(
        rule=fwh9_source_disjunct_equations)

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _solar_salt_ohtc_calculation(m)
    _create_arcs(m)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.discharge)
    return m


def _solar_salt_ohtc_calculation(m):
    """Block of equations for computing heat_transfer coefficient
    """

    # m = disj.model()

    # Calculate heat transfer coefficient for solar salt heat
    # exchanger
    m.fs.discharge.data_hxd_solar = {
        'tube_thickness': 0.004,
        'tube_inner_dia': 0.032,
        'tube_outer_dia': 0.036,
        'k_steel': 21.5,
        'number_tubes': 20,
        'shell_inner_dia': 1
    }

    # Data to compute overall heat transfer coefficient for the charge
    # heat exchanger using the Sieder-Tate Correlation. Parameters for
    # tube diameter and thickness assumed from the data in (2017) He
    # et al., Energy Procedia 105, 980-985
    m.fs.discharge.tube_thickness = Param(
        initialize=m.fs.discharge.data_hxd_solar['tube_thickness'],
        doc='Tube thickness [m]')
    m.fs.discharge.hxd.tube_inner_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['tube_inner_dia'],
        doc='Tube inner diameter [m]')
    m.fs.discharge.hxd.tube_outer_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['tube_outer_dia'],
        doc='Tube outer diameter [m]')
    m.fs.discharge.hxd.k_steel = Param(
        initialize=m.fs.discharge.data_hxd_solar['k_steel'],
        doc='Thermal conductivity of steel [W/mK]')
    m.fs.discharge.hxd.n_tubes = Param(
        initialize=m.fs.discharge.data_hxd_solar['number_tubes'],
        doc='Number of tubes')
    m.fs.discharge.hxd.shell_inner_dia = Param(
        initialize=m.fs.discharge.data_hxd_solar['shell_inner_dia'],
        doc='Shell inner diameter [m]')

    # Calculate Reynolds, Prandtl, and Nusselt number for the salt and
    # steam side of discharge heat exchanger
    m.fs.discharge.hxd.tube_cs_area = Expression(
        expr=(pi / 4) *
        (m.fs.discharge.hxd.tube_inner_dia ** 2),
        doc="Tube cross sectional area")
    m.fs.discharge.hxd.tube_out_area = Expression(
        expr=(pi / 4) *
        (m.fs.discharge.hxd.tube_outer_dia ** 2),
        doc="Tube cross sectional area including thickness [m2]")
    m.fs.discharge.hxd.shell_eff_area = Expression(
        expr=(
            (pi / 4) *
            (m.fs.discharge.hxd.shell_inner_dia ** 2) -
            m.fs.discharge.hxd.n_tubes *
            m.fs.discharge.hxd.tube_out_area),
        doc="Effective shell cross sectional area [m2]")

    # Calculate Reynolds number for the salt
    m.fs.discharge.hxd.salt_reynolds_number = Expression(
        expr=(
            (m.fs.discharge.hxd.inlet_1.flow_mass[0] *
             m.fs.discharge.hxd.tube_outer_dia) /
            (m.fs.discharge.hxd.shell_eff_area *
             m.fs.discharge.hxd.side_1.
             properties_in[0].dynamic_viscosity["Liq"])
        ),
        doc="Salt Reynolds Number")

    # Calculate Prandtl number for the salt
    m.fs.discharge.hxd.salt_prandtl_number = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.
            properties_in[0].cp_specific_heat["Liq"] *
            m.fs.discharge.hxd.side_1.
            properties_in[0].dynamic_viscosity["Liq"] /
            m.fs.discharge.hxd.side_1.
            properties_in[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number")

    # Calculate Prandtl Wall number for the salt
    m.fs.discharge.hxd.salt_prandtl_wall = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.
            properties_out[0].cp_specific_heat["Liq"] *
            m.fs.discharge.hxd.side_1.
            properties_out[0].dynamic_viscosity["Liq"] /
            m.fs.discharge.hxd.side_1.
            properties_out[0].thermal_conductivity["Liq"]
        ),
        doc="Salt Prandtl Number at wall")

    # Calculate Nusselt number for the salt
    m.fs.discharge.hxd.salt_nusselt_number = Expression(
        expr=(
            0.35 *
            (m.fs.discharge.hxd.salt_reynolds_number**0.6) *
            (m.fs.discharge.hxd.salt_prandtl_number**0.4) *
            ((m.fs.discharge.hxd.salt_prandtl_number /
              m.fs.discharge.hxd.salt_prandtl_wall) ** 0.25) *
            (2**0.2)
        ),
        doc="Salt Nusslet Number from 2019, App Ener (233-234), 126")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_reynolds_number = Expression(
        expr=(
            m.fs.discharge.hxd.inlet_2.flow_mol[0] *
            m.fs.discharge.hxd.side_2.properties_in[0].mw *
            m.fs.discharge.hxd.tube_inner_dia /
            (m.fs.discharge.hxd.tube_cs_area *
             m.fs.discharge.hxd.n_tubes *
             m.fs.discharge.hxd.side_2.
             properties_in[0].visc_d_phase["Vap"])
        ),
        doc="Steam Reynolds Number")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_prandtl_number = Expression(
        expr=(
            (m.fs.discharge.hxd.side_2.
             properties_in[0].cp_mol /
             m.fs.discharge.hxd.side_2.
             properties_in[0].mw) *
            m.fs.discharge.hxd.side_2.
            properties_in[0].visc_d_phase["Vap"] /
            m.fs.discharge.hxd.side_2.
            properties_in[0].therm_cond_phase["Vap"]
        ),
        doc="Steam Prandtl Number")

    # Calculate Reynolds number for the steam
    m.fs.discharge.hxd.steam_nusselt_number = Expression(
        expr=(
            0.023 *
            (m.fs.discharge.
             hxd.steam_reynolds_number**0.8) *
            (m.fs.discharge.
             hxd.steam_prandtl_number**(0.33)) *
            ((m.fs.discharge.hxd.
              side_2.properties_in[0].visc_d_phase["Vap"] /
              m.fs.discharge.hxd.side_2.
              properties_out[0].visc_d_phase["Liq"]) ** 0.14)
        ),
        doc="Steam Nusslet Number from 2001 Zavoico, Sandia")

    # Calculate heat transfer coefficients for the salt and steam
    # sides of discharge heat exchanger
    m.fs.discharge.hxd.h_salt = Expression(
        expr=(
            m.fs.discharge.hxd.side_1.
            properties_in[0].thermal_conductivity["Liq"] *
            m.fs.discharge.hxd.salt_nusselt_number /
            m.fs.discharge.hxd.tube_outer_dia
        ),
        doc="Salt side convective heat transfer coefficient [W/mK]")
    m.fs.discharge.hxd.h_steam = Expression(
        expr=(
            m.fs.discharge.hxd.side_2.
            properties_in[0].therm_cond_phase["Vap"] *
            m.fs.discharge.hxd.steam_nusselt_number /
            m.fs.discharge.hxd.tube_inner_dia
        ),
        doc="Steam side convective heat transfer coefficient [W/mK]")

    # Rewrite overall heat transfer coefficient constraint to avoid
    # denominators
    m.fs.discharge.hxd.tube_dia_ratio = (
        m.fs.discharge.hxd.tube_outer_dia /
        m.fs.discharge.hxd.tube_inner_dia)
    m.fs.discharge.hxd.log_tube_dia_ratio = log(
        m.fs.discharge.hxd.tube_dia_ratio)

    @m.fs.discharge.hxd.Constraint(m.fs.time)
    def constraint_hxd_ohtc(b, t):
        # return (
        #     m.fs.discharge.hxd.overall_heat_transfer_coefficient[t]
        #     == 1 / ((1 / m.fs.discharge.hxd.h_salt)
        #             + ((m.fs.discharge.hxd.tube_outer_dia *
        #                 m.fs.discharge.hxd.log_tube_dia_ratio) /
        #                 (2 * m.fs.discharge.hxd.k_steel))
        #             + (m.fs.discharge.hxd.tube_dia_ratio /
        #                m.fs.discharge.hxd.h_steam))
        # )
        # ------ modified by esrawli: equation rewritten to avoid denominators
        return (
            m.fs.discharge.hxd.
            overall_heat_transfer_coefficient[t] *
            (2 * m.fs.discharge.hxd.k_steel *
             m.fs.discharge.hxd.h_steam +
             m.fs.discharge.hxd.tube_outer_dia *
             m.fs.discharge.hxd.log_tube_dia_ratio *
             m.fs.discharge.hxd.h_salt *
             m.fs.discharge.hxd.h_steam +
             m.fs.discharge.hxd.tube_dia_ratio *
             m.fs.discharge.hxd.h_salt *
             2 * m.fs.discharge.hxd.k_steel)
        ) == (2 * m.fs.discharge.hxd.k_steel *
              m.fs.discharge.hxd.h_salt *
              m.fs.discharge.hxd.h_steam)


def _make_constraints(m):
    """Create arcs"""

    @m.fs.discharge.es_turbine.Constraint(
        m.fs.time,
        doc="Turbine outlet should be a saturated steam")
    def constraint_esturbine_temperature_out(b, t):
        return (
            b.control_volume.properties_out[t].temperature ==
            b.control_volume.properties_out[t].temperature_sat
        )


def _create_arcs(m):
    """Create arcs"""

    m.fs.discharge.essplit_to_hxd = Arc(
        source=m.fs.discharge.es_split.to_hxd,
        destination=m.fs.discharge.hxd.inlet_2,
        doc="Connection from ES splitter to HXD"
    )
    m.fs.discharge.hxd_to_esturbine = Arc(
        source=m.fs.discharge.hxd.outlet_2,
        destination=m.fs.discharge.es_turbine.inlet,
        doc="Connection from HXD to ES turbine"
    )


def disconect_arcs(m):
    """Create arcs"""

    # Disconnect arcs from ultra supercritical plant base model to
    # connect the charge heat exchanger and disjuncts
    for arc_s in [m.fs.condpump_to_fwh1,
                  m.fs.fwh4_to_fwh5,
                  m.fs.booster_to_fwh6,
                  m.fs.bfp_to_fwh8,
                  m.fs.fwh9_to_boiler]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()


def add_disjunction(m):
    """Add water source disjunction to the
    model
    """

    # Add disjunction 1 for ccs source steam selection
    m.fs.hxd_source_disjunction = Disjunction(
        expr=[
            m.fs.discharge.booster_source_disjunct,
            m.fs.discharge.bfp_source_disjunct,
            m.fs.discharge.fwh4_source_disjunct,
            m.fs.discharge.fwh9_source_disjunct,
            m.fs.discharge.condpump_source_disjunct
            ]
    )

    # Expand arcs within the disjuncts
    expand_arcs.obj_iter_kwds['descend_into'] = (Block, Disjunct)
    TransformationFactory("network.expand_arcs").apply_to(m.fs.discharge)

    return m


def condpump_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Condenser Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.condpump_source_disjunct.condpump_to_essplit = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from Condenser pump to ES splitter"
    )
    m.fs.discharge.condpump_source_disjunct.essplit_to_fwh1 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[1].inlet_2,
        doc="Connection from ES splitter to FWH1"
    )

    m.fs.discharge.condpump_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2
    )

    m.fs.discharge.condpump_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2
    )

    m.fs.discharge.condpump_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2
    )

    m.fs.discharge.condpump_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh4_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH4 Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh4_source_disjunct.fwh4_to_essplit = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH4 to ES splitter"
    )
    m.fs.discharge.fwh4_source_disjunct.essplit_to_fwh5 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[5].inlet_2,
        doc="Connection from ES splitter to FWH5"
    )

    m.fs.discharge.fwh4_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )

    m.fs.discharge.fwh4_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2
    )

    m.fs.discharge.fwh4_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2
    )

    m.fs.discharge.fwh4_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def booster_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Booster Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.booster_source_disjunct.booster_to_essplit = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from Booster pump to ES splitter"
    )
    m.fs.discharge.booster_source_disjunct.essplit_to_fwh6 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[6].inlet_2,
        doc="Connection from ES splitter to FWH6"
    )

    m.fs.discharge.booster_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2
    )

    m.fs.discharge.booster_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )

    m.fs.discharge.booster_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2
    )

    m.fs.discharge.booster_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def bfp_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from Boiler Feed Pump
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.bfp_source_disjunct.bfp_to_essplit = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from BFP to ES splitter"
    )
    m.fs.discharge.bfp_source_disjunct.essplit_to_fwh8 = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.fwh[8].inlet_2,
        doc="Connection from ES splitter to FWH8"
    )

    m.fs.discharge.bfp_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2
    )

    m.fs.discharge.bfp_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )

    m.fs.discharge.bfp_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2
    )

    m.fs.discharge.bfp_source_disjunct.fwh9_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.boiler.inlet
    )


def fwh9_source_disjunct_equations(disj):
    """Disjunction 1: Water is sourced from FWH9
    """

    m = disj.model()

    # Define arcs to connect units within disjunct
    m.fs.discharge.fwh9_source_disjunct.fwh9_to_essplit = Arc(
        source=m.fs.fwh[9].outlet_2,
        destination=m.fs.discharge.es_split.inlet,
        doc="Connection from FWH9 to the ES SPlitter"
    )
    m.fs.discharge.fwh9_source_disjunct.essplit_to_boiler = Arc(
        source=m.fs.discharge.es_split.to_fwh,
        destination=m.fs.boiler.inlet,
        doc="Connection from ES splitter to Boiler"
    )

    m.fs.discharge.fwh9_source_disjunct.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2,
        destination=m.fs.fwh[5].inlet_2
    )

    m.fs.discharge.fwh9_source_disjunct.condpump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet,
        destination=m.fs.fwh[1].inlet_2
    )

    m.fs.discharge.fwh9_source_disjunct.booster_to_fwh6 = Arc(
        source=m.fs.booster.outlet,
        destination=m.fs.fwh[6].inlet_2
    )

    m.fs.discharge.fwh9_source_disjunct.bfp_to_fwh8 = Arc(
        source=m.fs.bfp.outlet,
        destination=m.fs.fwh[8].inlet_2
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
    m.fs.discharge.hxd.area.fix(2000)  # m2
    # m.fs.discharge.hxd.overall_heat_transfer_coefficient.fix(1000)

    # Define storage fluid conditions. The fluid inlet flow is fixed
    # during initialization, but is unfixed and determined during
    # optimization
    m.fs.discharge.hxd.inlet_1.flow_mass.fix(200)   # kg/s
    m.fs.discharge.hxd.inlet_1.temperature.fix(831.15)  # K
    m.fs.discharge.hxd.inlet_1.pressure.fix(101325)  # Pa

    m.fs.discharge.es_split.inlet.flow_mol.fix(17854)   # kg/s
    m.fs.discharge.es_split.inlet.enth_mol.fix(52232)  # K
    m.fs.discharge.es_split.inlet.pressure.fix(3.4958e+07)  # Pa

    m.fs.discharge.es_split.split_fraction[0, "to_hxd"].fix(0.2)

    # m.fs.discharge.es_turbine.ratioP.fix(0.0286)
    m.fs.discharge.es_turbine.constraint_esturbine_temperature_out.deactivate()
    m.fs.discharge.es_turbine.outlet.pressure.fix(6896)
    m.fs.discharge.es_turbine.efficiency_isentropic.fix(0.8)


def set_scaling_factors(m):
    """Scaling factors in the flowsheet

    """

    # Include scaling factors for solar, hitec, and thermal oil charge
    # heat exchangers
    for htf in [m.fs.discharge.hxd]:
        iscale.set_scaling_factor(htf.area, 1e-2)
        iscale.set_scaling_factor(
            htf.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(htf.tube.heat, 1e-5)
        iscale.set_scaling_factor(htf.shell.heat, 1e-5)
        iscale.set_scaling_factor(htf.tube.properties_in[0].flow_mol, 1e-3)
        iscale.set_scaling_factor(htf.shell.properties_in[0].flow_mass, 1e-2)
        iscale.set_scaling_factor(htf.tube.properties_out[0].flow_mol, 1e-3)
        iscale.set_scaling_factor(htf.shell.properties_out[0].flow_mass, 1e-2)
        iscale.set_scaling_factor(htf.tube.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(htf.shell.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(htf.tube.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(htf.shell.properties_out[0].pressure, 1e-6)

    for k in m.set_turbine:
        a = m.fs.turbine[k].control_volume
        iscale.set_scaling_factor(a.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(a.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(a.properties_in[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(a.properties_out[0].enth_mol, 1e-5)

    for k in m.set_fwh:
        b = m.fs.fwh[k]
        iscale.set_scaling_factor(b.shell.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(b.shell.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(b.tube.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(b.tube.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(b.shell.properties_in[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(b.shell.properties_out[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(b.tube.properties_in[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(b.tube.properties_out[0].enth_mol, 1e-5)

    for k in m.set_fwh_mixer:
        c = m.fs.fwh_mixer[k]
        iscale.set_scaling_factor(c.steam_state[0].pressure, 1e-6)
        iscale.set_scaling_factor(c.drain_state[0].pressure, 1e-6)
        iscale.set_scaling_factor(c.mixed_state[0].pressure, 1e-6)
        iscale.set_scaling_factor(c.steam_state[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(c.drain_state[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(c.mixed_state[0].enth_mol, 1e-5)

    for j in [m.fs.boiler, m.fs.reheater[1],
              m.fs.reheater[2], m.fs.bfp, m.fs.bfpt]:
        iscale.set_scaling_factor(
            j.control_volume.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(
            j.control_volume.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(
            j.control_volume.properties_in[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(
            j.control_volume.properties_out[0].enth_mol, 1e-5)

    # # scaling factors for CCS: to be added later
    # bfs = m.fs.boiler_fireside
    # iscale.set_scaling_factor(bfs.waterwall_heat[0, 1], 1e-6)
    # iscale.set_scaling_factor(bfs.platen_heat[0], 1e-6)
    # iscale.set_scaling_factor(bfs.roof_heat[0], 1e-6)
    # iscale.set_scaling_factor(m.fs.ccs_reformer.control_volume.heat, 1e-6)

    for est in [m.fs.discharge.es_turbine.control_volume]:
        iscale.set_scaling_factor(est.properties_in[0].pressure, 1e-6)
        iscale.set_scaling_factor(est.properties_out[0].pressure, 1e-6)
        iscale.set_scaling_factor(est.properties_in[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(est.properties_out[0].enth_mol, 1e-5)
        iscale.set_scaling_factor(est.work, 1e-6)


def initialize(m, solver=None,
               outlvl=idaeslog.NOTSET,
               optarg={"tol": 1e-8,
                       "max_iter": 300},
               fluid=None,
               source=None):
    """Initialize the units included in the discharge model
    """

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    # Include scaling factors
    iscale.calculate_scaling_factors(m)

    # Initialize splitters
    m.fs.discharge.es_split.initialize(outlvl=outlvl,
                                       optarg=solver.options)

    propagate_state(m.fs.discharge.essplit_to_hxd)
    m.fs.discharge.hxd.initialize(outlvl=outlvl,
                                  optarg=solver.options)

    propagate_state(m.fs.discharge.hxd_to_esturbine)
    m.fs.discharge.es_turbine.initialize(outlvl=outlvl,
                                         optarg=solver.options)
    m.fs.discharge.es_turbine.constraint_esturbine_temperature_out.activate()
    m.fs.discharge.es_turbine.outlet.pressure.unfix()
    print('DOFs before init solution =', degrees_of_freedom(m))
    res = solver.solve(m,
                       tee=False,
                       options=optarg)

    print("Discharge Model Initialization = ",
          res.solver.termination_condition)
    print("*************   Discharge Model Initialized   ******************")


def build_costing(m, solver=None, optarg={"tol": 1e-8, "max_iter": 300}):
    """ Add cost correlations for the storage design analysis. This
    function is used to estimate the capital and operating cost of
    integrating an energy storage system. It contains cost
    correlations to estimate the capital cost of discharge heat
    exchanger, and molten salt pump.

    """

    # All the computed capital costs are annualized. The operating
    # cost is for 1 year. In addition, operating savings in terms of
    # annual coal cost are estimated based on the differential
    # reduction of coal consumption as compared to ramped baseline
    # power plant. Unless other wise stated, the cost correlations
    # used here (except IDAES costing method) are taken from 2nd
    # Edition, Product & Process Design Principles, Seider et al.

    ###########################################################################
    #  Data                                                                   #
    ###########################################################################
    m.CE_index = 607.5  # Chemical engineering cost index for 2019

    # The q baseline_charge corresponds to heat duty of a plant with
    # no storage and producing 400 MW power
    m.data_cost = {
        'coal_price': 2.11e-9,
        'cooling_price': 3.3e-9,
        'q_baseline_charge': 838565942.4732262,
        'solar_salt_price': 0.49,
        'hitec_salt_price': 0.93,
        'thermal_oil_price': 6.72,  # $/kg
        'storage_tank_material': 3.5,
        'storage_tank_insulation': 235,
        'storage_tank_foundation': 1210
    }
    m.data_salt_pump = {
        'FT': 1.5,
        'FM': 2.0,
        'head': 3.281*5,
        'motor_FT': 1,
        'nm': 1
    }
    m.data_storage_tank = {
        'LbyD': 0.325,
        'tank_thickness': 0.039,
        'material_density': 7800
    }

    # Main flowsheet operation data
    m.fs.discharge.coal_price = Param(
        initialize=m.data_cost['coal_price'],
        doc='Coal price based on HHV for Illinois No.6 (NETL Report) $/J')

    ###########################################################################
    #  Operating hours                                                        #
    ###########################################################################
    m.number_hours_per_day = 6
    m.number_of_years = 30

    m.fs.discharge.hours_per_day = Var(
        initialize=m.number_hours_per_day,
        bounds=(0, 12),
        doc='Estimated number of hours of charging per day'
    )

    # Fix number of hours of discharging to 6
    m.fs.discharge.hours_per_day.fix(m.number_hours_per_day)

    # Define number of years over which the capital cost is annualized
    m.fs.discharge.num_of_years = Param(
        initialize=m.number_of_years,
        doc='Number of years for capital cost annualization')

    ###########################################################################
    #  Capital cost                                                           #
    ###########################################################################

    m.fs.discharge.hxd.get_costing()
    m.fs.discharge.hxd.costing.CE_index = m.CE_index
    # Initialize Solar and Hitec charge heat exchanger costing
    # correlations
    icost.initialize(m.fs.discharge.hxd.costing)

    # --------------------------------------------
    #  Salt-pump costing
    # --------------------------------------------
    # The salt pump is not explicitly modeled. Thus, the IDAES cost
    # method is not used for this equipment at this time.  The primary
    # purpose of the salt pump is to move molten salt and not to
    # change the pressure. Thus the pressure head is computed assuming
    # that the salt is moved on an average of 5m linear distance.
    m.fs.discharge.spump_FT = pyo.Param(
        initialize=m.data_salt_pump['FT'],
        doc='Pump Type Factor for vertical split case')
    m.fs.discharge.spump_FM = pyo.Param(
        initialize=m.data_salt_pump['FM'],
        doc='Pump Material Factor Stainless Steel')
    m.fs.discharge.spump_head = pyo.Param(
        initialize=m.data_salt_pump['head'],
        doc='Pump Head 5m in Ft.')
    m.fs.discharge.spump_motorFT = pyo.Param(
        initialize=m.data_salt_pump['motor_FT'],
        doc='Motor Shaft Type Factor')
    m.fs.discharge.spump_nm = pyo.Param(
        initialize=m.data_salt_pump['nm'],
        doc='Motor Shaft Type Factor')

    #  Solar salt-pump costing
    m.fs.discharge.spump_Qgpm = pyo.Expression(
        expr=(m.fs.discharge.hxd.
              side_1.properties_in[0].flow_mass *
              264.17 * 60 /
              (m.fs.discharge.hxd.
               side_1.properties_in[0].density["Liq"])),
        doc="Conversion of solar salt flow mass to vol flow [gal per min]"
    )
    m.fs.discharge.dens_lbft3 = pyo.Expression(
        expr=m.fs.discharge.hxd.side_1.properties_in[0].
        density["Liq"] * 0.062428,
        doc="pump size factor"
    )  # density in lb per ft3
    m.fs.discharge.spump_sf = pyo.Expression(
        expr=(m.fs.discharge.spump_Qgpm
              * (m.fs.discharge.spump_head ** 0.5)),
        doc="Pump size factor"
    )
    m.fs.discharge.pump_CP = pyo.Expression(
        expr=(
            m.fs.discharge.spump_FT * m.fs.discharge.spump_FM *
            exp(
                9.2951
                - 0.6019 * log(m.fs.discharge.spump_sf)
                + 0.0519 * ((log(m.fs.discharge.spump_sf))**2)
            )
        ),
        doc="Salt pump base (purchase) cost in $"
    )
    # Costing motor
    m.fs.discharge.spump_np = pyo.Expression(
        expr=(
            -0.316
            + 0.24015 * log(m.fs.discharge.spump_Qgpm)
            - 0.01199 * ((log(m.fs.discharge.spump_Qgpm))**2)
        ),
        doc="Fractional efficiency of the pump horse power"
    )
    m.fs.discharge.motor_pc = pyo.Expression(
        expr=(
            (m.fs.discharge.spump_Qgpm *
             m.fs.discharge.spump_head *
             m.fs.discharge.dens_lbft3) /
            (33000 * m.fs.discharge.spump_np *
             m.fs.discharge.spump_nm)
        ),
        doc="Motor power consumption in horsepower"
    )
    # Motor purchase cost
    log_motor_pc = log(m.fs.discharge.motor_pc)
    m.fs.discharge.motor_CP = pyo.Expression(
        expr=(
            m.fs.discharge.spump_motorFT *
            exp(
                5.4866
                + 0.13141 * log_motor_pc
                + 0.053255 * (log_motor_pc**2)
                + 0.028628 * (log_motor_pc**3)
                - 0.0035549 * (log_motor_pc**4)
            )
        ),
        doc="Salt Pump's Motor Base Cost in $"
    )
    # Pump and motor purchase cost pump
    m.fs.discharge.spump_purchase_cost = pyo.Var(
        initialize=100000,
        bounds=(0, None),
        doc="Salt pump and motor purchase cost in $"
    )

    def solar_spump_purchase_cost_rule(b):
        return (
            m.fs.discharge.spump_purchase_cost == (
                m.fs.discharge.pump_CP
                + m.fs.discharge.motor_CP) *
            (m.CE_index / 394)
        )
    m.fs.discharge.spump_purchase_cost_eq = pyo.Constraint(
        rule=solar_spump_purchase_cost_rule)

    # Initialize cost correlation
    calculate_variable_from_constraint(
        m.fs.discharge.spump_purchase_cost,
        m.fs.discharge.spump_purchase_cost_eq)

    # --------------------------------------------
    # Total annualized capital cost for solar salt
    # --------------------------------------------
    # Capital cost var at flowsheet level to handle the salt capital
    # cost depending on the salt selected.
    m.fs.discharge.capital_cost = pyo.Var(
        initialize=1000000,
        doc="Annualized capital cost")

    # Annualize capital cost for the solar salt
    def solar_cap_cost_rule(b):
        return (m.fs.discharge.capital_cost * m.fs.discharge.num_of_years ==
                m.fs.discharge.spump_purchase_cost
                + m.fs.discharge.hxd.costing.purchase_cost)
    m.fs.discharge.cap_cost_eq = pyo.Constraint(
        rule=solar_cap_cost_rule)

    # Initialize capital cost
    calculate_variable_from_constraint(
        m.fs.discharge.capital_cost,
        m.fs.discharge.cap_cost_eq)

    ###########################################################################
    #  Annual operating cost
    ###########################################################################
    m.fs.discharge.operating_hours = pyo.Expression(
        expr=365 * 3600 * m.fs.discharge.hours_per_day,
        doc="Number of operating hours per year")
    m.fs.discharge.operating_cost = pyo.Var(
        initialize=1000000,
        # -------- modified by esrawli
        # bounds=(1e-5, None),
        bounds=(0, None),
        # --------
        doc="Operating cost")  # add units

    def op_cost_rule(b):
        return m.fs.discharge.operating_cost == (
            m.fs.discharge.operating_hours * m.fs.discharge.coal_price *
            m.fs.plant_heat_duty[0] * 1e6
        )
    m.fs.discharge.op_cost_eq = pyo.Constraint(rule=op_cost_rule)

    # Initialize operating cost
    calculate_variable_from_constraint(
        m.fs.discharge.operating_cost,
        m.fs.discharge.op_cost_eq)

    res = solver.solve(m,
                       tee=False,
                       symbolic_solver_labels=True,
                       options=optarg)
    print("Cost Initialization = ",
          res.solver.termination_condition)
    print("******************** Costing Initialized *************************")
    print('')
    print('')


def add_bounds(m, source=None):
    """Add bounds to units in discharge model

    """

    # Unless stated otherwise, the temperature is in K, pressure in
    # Pa, flow in mol/s, massic flow in kg/s, and heat and heat duty
    # in W

    m.flow_max = m.main_flow * 3.2  # in mol/s
    m.salt_flow_max = 1200  # in kg/s
    m.heat_duty_bound = 200e6

    # Charge heat exchanger section
    for hxd in [m.fs.discharge.hxd]:
        hxd.inlet_2.flow_mol.setlb(0)
        hxd.inlet_2.flow_mol.setub(0.2 * m.flow_max)
        hxd.inlet_1.flow_mass.setlb(0)
        hxd.inlet_1.flow_mass.setub(m.salt_flow_max)
        hxd.outlet_2.flow_mol.setlb(0)
        hxd.outlet_2.flow_mol.setub(0.2 * m.flow_max)
        hxd.outlet_1.flow_mass.setlb(0)
        hxd.outlet_1.flow_mass.setub(m.salt_flow_max)
        hxd.inlet_1.pressure.setlb(101320)
        hxd.inlet_1.pressure.setub(101330)
        hxd.outlet_1.pressure.setlb(101320)
        hxd.outlet_1.pressure.setub(101330)
        hxd.heat_duty.setlb(0)
        hxd.heat_duty.setub(m.heat_duty_bound)
        hxd.shell.heat.setlb(-m.heat_duty_bound)
        hxd.shell.heat.setub(0)
        hxd.tube.heat.setlb(0)
        hxd.tube.heat.setub(m.heat_duty_bound)
        hxd.shell.properties_in[0].enthalpy_mass.setlb(0)
        hxd.shell.properties_in[0].\
            enthalpy_mass.setub(1.5e6)
        hxd.shell.properties_out[0].enthalpy_mass.setlb(0)
        hxd.shell.properties_out[0].\
            enthalpy_mass.setub(1.5e6)
        hxd.overall_heat_transfer_coefficient.setlb(0)
        hxd.overall_heat_transfer_coefficient.setub(10000)
        hxd.area.setlb(0)
        hxd.area.setub(5000)  # TODO: Check this value
        hxd.costing.pressure_factor.setlb(0)  # no unit
        hxd.costing.pressure_factor.setub(1e5)  # no unit
        hxd.costing.purchase_cost.setlb(0)  # no unit
        hxd.costing.purchase_cost.setub(1e7)  # no unit
        hxd.costing.base_cost_per_unit.setlb(0)
        hxd.costing.base_cost_per_unit.setub(1e6)
        hxd.costing.material_factor.setlb(0)
        hxd.costing.material_factor.setub(10)
        hxd.delta_temperature_in.setlb(10)
        hxd.delta_temperature_out.setlb(9.4)
        hxd.delta_temperature_in.setub(299)  # 80
        hxd.delta_temperature_out.setub(499)  # 79.9

    # Add bounds to cost-related terms
    m.fs.discharge.capital_cost.setlb(0)  # no units
    m.fs.discharge.capital_cost.setub(1e7)

    for salt_cost in [m.fs.discharge]:
        salt_cost.capital_cost.setlb(0)
        salt_cost.capital_cost.setub(1e7)
        salt_cost.spump_purchase_cost.setlb(0)
        salt_cost.spump_purchase_cost.setub(1e7)

    # Add bounds needed in VHP and HP source disjuncts
    for split in [m.fs.discharge.es_split]:
        split.to_hxd.flow_mol[:].setlb(0)
        split.to_hxd.flow_mol[:].setub(0.2 * m.flow_max)
        split.to_fwh.flow_mol[:].setlb(0)
        split.to_fwh.flow_mol[:].setub(0.5 * m.flow_max)
        split.split_fraction[0.0, "to_hxd"].setlb(0)
        split.split_fraction[0.0, "to_hxd"].setub(1)
        split.split_fraction[0.0, "to_fwh"].setlb(0)
        split.split_fraction[0.0, "to_fwh"].setub(1)
        split.inlet.flow_mol[:].setlb(0)
        split.inlet.flow_mol[:].setub(m.flow_max)

    m.fs.plant_power_out[0].setlb(300)
    m.fs.plant_power_out[0].setub(700)

    for unit_k in [m.fs.booster]:
        unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
        unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
        unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s
        # unit_k.control_volume.work.setlb(0)
        # unit_k.control_volume.work.setub(1e8)
        # unit_k.control_volume.deltaP.setlb(0)
        # unit_k.control_volume.deltaP.setub(1e8)
        # unit_k.efficiency_isentropic.setlb(0)
        # unit_k.efficiency_isentropic.setub(1)
        # unit_k.ratioP.setlb(0)
        # unit_k.ratioP.setub(10)

    for k in m.set_turbine:
        m.fs.turbine[k].work.setlb(-1e10)
        m.fs.turbine[k].work.setub(0)
        # m.fs.turbine[k].control_volume.deltaP.setlb(-1e8)
        # m.fs.turbine[k].control_volume.deltaP.setub(0)
        # m.fs.turbine[k].efficiency_isentropic.setlb(0)
        # m.fs.turbine[k].efficiency_isentropic.setub(1)
        # m.fs.turbine[k].ratioP.setlb(0)
        # m.fs.turbine[k].ratioP.setub(1)
        # m.fs.turbine[k].efficiency_mech.setlb(0)
        # m.fs.turbine[k].efficiency_mech.setub(1)
        # m.fs.turbine[k].shaft_speed.setlb(0)
        # m.fs.turbine[k].shaft_speed.setub(100)

    # for unit_k in [m.fs.bfp, m.fs.cond_pump]:
    #     unit_k.inlet.flow_mol[:].setlb(0)  # mol/s
    #     unit_k.inlet.flow_mol[:].setub(m.flow_max)  # mol/s
    #     unit_k.outlet.flow_mol[:].setlb(0)  # mol/s
    #     unit_k.outlet.flow_mol[:].setub(m.flow_max)  # mol/s
    #     unit_k.control_volume.work.setlb(0)
    #     unit_k.control_volume.work.setub(1e9)
    #     unit_k.control_volume.deltaP.setlb(0)
    #     unit_k.control_volume.deltaP.setub(1e9)
    #     unit_k.efficiency_isentropic.setlb(0)
    #     unit_k.efficiency_isentropic.setub(1)
    #     unit_k.ratioP.setlb(0)
    #     unit_k.ratioP.setub(1000)

    # m.fs.bfpt.work.setlb(-1e10)
    # m.fs.bfpt.work.setub(0)
    # m.fs.bfpt.control_volume.deltaP.setlb(-1e8)
    # m.fs.bfpt.control_volume.deltaP.setub(0)
    # m.fs.bfpt.efficiency_isentropic.setlb(0)
    # m.fs.bfpt.efficiency_isentropic.setub(1)
    # m.fs.bfpt.ratioP.setlb(0)
    # m.fs.bfpt.ratioP.setub(1)
    # m.fs.bfpt.efficiency_mech.setlb(0)
    # m.fs.bfpt.efficiency_mech.setub(1)
    # m.fs.bfpt.shaft_speed.setlb(0)
    # m.fs.bfpt.shaft_speed.setub(100)

    # m.fs.plant_heat_duty[0].setlb(500)
    # m.fs.plant_heat_duty[0].setub(2000)

    # m.fs.discharge.operating_cost.setlb(0)
    # m.fs.discharge.operating_cost.setub(1e9)

    # # Adding bounds on turbine splitters flow
    # for k in m.set_turbine_splitter:
    #     ts = m.fs.turbine_splitter[k]
    #     ts.inlet.flow_mol[:].setlb(0)
    #     ts.inlet.flow_mol[:].setub(m.flow_max)
    #     ts.outlet_1.flow_mol[:].setlb(0)
    #     ts.outlet_1.flow_mol[:].setub(m.flow_max)
    #     ts.outlet_2.flow_mol[:].setlb(0)
    #     ts.outlet_2.flow_mol[:].setub(m.flow_max)
    # m.fs.turbine_splitter[6].outlet_3.flow_mol[:].setlb(0)
    # m.fs.turbine_splitter[6].outlet_3.flow_mol[:].setub(m.flow_max)
    # m.fs.turbine_splitter[6].split_fraction[0, "outlet_3"].setlb(0)
    # m.fs.turbine_splitter[6].split_fraction[0, "outlet_3"].setub(1)

    # for unit_b in [m.fs.boiler, m.fs.reheater[1], m.fs.reheater[2]]:
    #     unit_b.control_volume.heat.setlb(0)
    #     unit_b.control_volume.heat.setub(1e10)
    #     unit_b.control_volume.deltaP.setlb(-1e8)
    #     unit_b.control_volume.deltaP.setub(0)

    # for j in m.set_fwh:
    #     fwh = m.fs.fwh[j]
    #     fwh.delta_temperature_in.setlb(1)
    #     fwh.delta_temperature_in.setub(500)
    #     fwh.delta_temperature_out.setlb(1)
    #     fwh.delta_temperature_out.setub(100)
    #     fwh.tube.heat[:].setlb(0)
    #     fwh.tube.heat[:].setub(1e9)
    #     fwh.shell.heat[:].setlb(-1e8)
    #     fwh.shell.heat[:].setub(0)
    #     fwh.tube.deltaP[:].setlb(-1e8)
    #     fwh.tube.deltaP[:].setub(0)
    #     fwh.shell.deltaP[:].setlb(-1e9)
    #     fwh.shell.deltaP[:].setub(0)

    # for p in m.set_fwh_mixer:
    #     fwhmix = m.fs.fwh_mixer[p]
    #     fwhmix.steam.flow_mol[:].setlb(0)
    #     fwhmix.steam.flow_mol[:].setub(m.flow_max)
    #     fwhmix.drain.flow_mol[:].setlb(0)
    #     fwhmix.drain.flow_mol[:].setub(m.flow_max)
    #     fwhmix.outlet.flow_mol[:].setlb(0)
    #     fwhmix.outlet.flow_mol[:].setub(m.flow_max)

    # m.fs.deaerator.steam.flow_mol[:].setlb(0)
    # m.fs.deaerator.steam.flow_mol[:].setub(m.flow_max)
    # m.fs.deaerator.drain.flow_mol[:].setlb(0)
    # m.fs.deaerator.drain.flow_mol[:].setub(m.flow_max)
    # m.fs.deaerator.feedwater.flow_mol[:].setlb(0)
    # m.fs.deaerator.feedwater.flow_mol[:].setub(m.flow_max)
    # # m.fs.deaerator.mixed_state[:].flow_mol.setlb(0)
    # # m.fs.deaerator.mixed_state[:].flow_mol.setub(m.flow_max)

    # m.fs.condenser_mix.main.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.main.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser_mix.bfpt.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.bfpt.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser_mix.drain.flow_mol[:].setlb(0)
    # m.fs.condenser_mix.drain.flow_mol[:].setub(m.flow_max)
    # # m.fs.condenser_mix.makeup.flow_mol[:].setlb(0)
    # # m.fs.condenser_mix.makeup.flow_mol[:].setub(m.flow_max)
    # # m.fs.condenser_mix.outlet.flow_mol[:].setlb(0)
    # # m.fs.condenser_mix.outlet.flow_mol[:].setub(m.flow_max)

    # m.fs.condenser.inlet.flow_mol[:].setlb(0)
    # m.fs.condenser.inlet.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser.outlet.flow_mol[:].setlb(0)
    # m.fs.condenser.outlet.flow_mol[:].setub(m.flow_max)
    # m.fs.condenser.control_volume.heat.setlb(-1e10)
    # m.fs.condenser.control_volume.heat.setub(0)

    return m


def main(m_usc, fluid=None, source=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_discharge_model(m_usc)

    # Give all the required inputs to the model
    set_model_input(m)

    # Add scaling factor
    set_scaling_factors(m)

    # Initialize the model with a sequential initialization
    print('DOF before initialization: ', degrees_of_freedom(m))
    initialize(m, fluid=fluid, source=source)
    print('DOF after initialization: ', degrees_of_freedom(m))

    # Add cost correlations
    build_costing(m, solver=solver)
    print('DOF after build costing: ', degrees_of_freedom(m))

    # Add bounds
    add_bounds(m, source=source)

    # Add disjunctions
    disconect_arcs(m)
    add_disjunction(m)

    return m, solver


def run_nlps(m,
             solver=None,
             source=None):
    """This function fixes the indicator variables of the disjuncts so to
    solve NLP problems

    """

    # Disjunction 1 for the steam source selection
    if source == "condpump":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh4":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "booster":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "bfp":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(True)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(False)
    elif source == "fwh9":
        m.fs.discharge.condpump_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh4_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.booster_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.bfp_source_disjunct.indicator_var.fix(False)
        m.fs.discharge.fwh9_source_disjunct.indicator_var.fix(True)
    else:
        print('Unrecognized source unit name!')

    TransformationFactory('gdp.fix_disjuncts').apply_to(m)
    print("The degrees of freedom after gdp transformation ",
          degrees_of_freedom(m))

    # strip_bounds = pyo.TransformationFactory('contrib.strip_var_bounds')
    # strip_bounds.apply_to(m, reversible=True)

    results = solver.solve(
        m,
        tee=True,
        symbolic_solver_labels=True,
        options={
            "linear_solver": "ma27",
            "max_iter": 150,
            # "bound_push": 1e-12,
            # "mu_init": 1e-8
        }
    )
    log_close_to_bounds(m)

    return m, results


def print_model(nlp_model, nlp_data):
    nlp = nlp_model.fs.discharge
    print('       ___________________________________________')
    if nlp.condpump_source_disjunct.binary_indicator_var.value == 1:
        print('        Disjunction 1: Water from Condenser Pump is sourced')
    elif (nlp.booster_source_disjunct.binary_indicator_var.value == 1):
        print('        Disjunction 1: Water from Booster Pump is sourced')
    elif (nlp.bfp_source_disjunct.binary_indicator_var.value == 1):
        print('        Disjunction 1: Water from Boiler Feed Pump is sourced')
    elif (nlp.fwh9_source_disjunct.binary_indicator_var.value == 1):
        print('        Disjunction 1: Water from FWH9 is sourced')
    elif (nlp.fwh4_source_disjunct.binary_indicator_var.value == 1):
        print('        Disjunction 1: Water from FWH4 is sourced')
    else:
        print('        Disjunction 1: Error')

    print('       ___________________________________________')
    print('')

    # log_infeasible_constraints(nlp_model)
    log_close_to_bounds(nlp_model)


def run_gdp(m):
    """Declare solver GDPopt and its options
    """

    opt = SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'RIC'  # LOA is an option
    # opt.CONFIG.OA_penalty_factor = 1e4
    # opt.CONFIG.max_slack = 1e4
    opt.CONFIG.call_after_subproblem_solve = print_model
    # opt.CONFIG.mip_solver = 'glpk'
    opt.CONFIG.mip_solver = 'cbc'
    # opt.CONFIG.mip_solver = 'gurobi_direct'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.tee = True
    opt.CONFIG.init_strategy = "no_init"
    opt.CONFIG.time_limit = "2400"

    results = opt.solve(
        m,
        tee=True,
        nlp_solver_args=dict(
            tee=True,
            symbolic_solver_labels=True,
            options={
                "linear_solver": "ma27",
                "max_iter": 150
            }
        )
    )

    return results


def print_results(m, results):

    print('================================')
    print("***************** Printing Results ******************")
    print('')
    print("Disjunctions")
    for d in m.component_data_objects(ctype=Disjunct,
                                      active=True,
                                      sort=True, descend_into=True):
        if abs(d.binary_indicator_var.value - 1) < 1e-6:
            print(d.name, ' should be selected!')
    print('')
    est = m.fs.discharge.es_turbine
    hxd = m.fs.discharge.hxd
    print('Obj (M$/year): {:.6f}'.format(value(m.obj) * 1e-6))
    print('Discharge capital cost ($/y): {:.6f}'.format(
        pyo.value(m.fs.discharge.capital_cost) * 1e-6))
    print('Charge Operating costs ($/y): {:.6f}'.format(
        pyo.value(m.fs.discharge.operating_cost) * 1e-6))
    print('Storage Turbine Power (MW): {:.6f}'.format(
        value(est.control_volume.work[0]) * -1e-6))
    print('Plant Power (MW): {:.6f}'.format(
        value(m.fs.plant_power_out[0])))
    print('Boiler feed water flow (mol/s): {:.6f}'.format(
        value(m.fs.boiler.inlet.flow_mol[0])))
    print('Boiler duty (MW_th): {:.6f}'.format(
        value((m.fs.boiler.heat_duty[0]
               + m.fs.reheater[1].heat_duty[0]
               + m.fs.reheater[2].heat_duty[0])
              * 1e-6)))
    print('')
    print('Source: Cond pump is selected!')
    print('Heat exchanger area (m2): {:.6f}'.format(
        value(hxd.area)))
    print('Heat exchanger cost ($/y): {:.6f}'.format(
        value(hxd.costing.purchase_cost / 15)))
    print('Salt flow (kg/s): {:.6f}'.format(
        value(hxd.inlet_1.flow_mass[0])))
    print('Salt temperature in (K): {:.6f}'.format(
        value(hxd.inlet_1.temperature[0])))
    print('Salt temperature out (K): {:.6f}'.format(
        value(hxd.outlet_1.temperature[0])))
    print('Steam flow to storage (mol/s): {:.6f}'.format(
        value(hxd.inlet_2.flow_mol[0])))
    print('Water temperature in (K): {:.6f}'.format(
        value(hxd.side_2.properties_in[0].temperature)))
    print('Steam temperature out (K): {:.6f}'.format(
        value(hxd.side_2.properties_out[0].temperature)))
    print('HXD overall heat transfer coefficient: {:.6f}'.format(
        value(hxd.overall_heat_transfer_coefficient[0])))
    print('Delta temperature at inlet (K): {:.6f}'.format(
        value(hxd.delta_temperature_in[0])))
    print('Delta temperature at outlet (K): {:.6f}'.format(
        value(hxd.delta_temperature_out[0])))
    print('Salt pump cost ($/y): {:.6f}'.format(
        value(m.fs.discharge.spump_purchase_cost)))
    print('')
    print('Condenser Pressure in (MPa): {:.6f}'.format(
        value(m.fs.condenser.inlet.pressure[0]) * 1e-6))
    print('Storage Turbine Pressure in (MPa): {:.6f}'.format(
        value(est.inlet.pressure[0]) * 1e-6))
    print('Storage Turbine Pressure out (MPa): {:.6f}'.format(
        value(est.outlet.pressure[0]) * 1e-6))
    print('Storage Turbine Vapor Frac in (K): {:.6f}'.format(
        value(est.control_volume.properties_in[0].vapor_frac)))
    print('Storage Turbine Temperature in (K): {:.6f}'.format(
        value(est.control_volume.properties_in[0].temperature)))
    print('Storage Turbine Temperature out (K): {:.6f}'.format(
        value(est.control_volume.properties_out[0].temperature)))
    print('Storage Turbine Saturation Temperature (K): {:.6f}'.format(
        value(est.control_volume.properties_out[0].temperature_sat)))
    print('Storage Split Fraction to HXD: {:.6f}'.format(
        value(m.fs.discharge.es_split.split_fraction[0, "to_hxd"])))
    print('')
    print('Salt density: {:.6f}'.format(
        value(hxd.side_1.properties_in[0].density['Liq'])))
    print('HXC heat duty: {:.6f}'.format(
        value(hxd.heat_duty[0]) * 1e-6))
    print('')
    print('Solver details')
    print(results)
    print(' ')
    print('==============================================================')
    # for v in m.component_data_objects(Var):
    #     if v.ub is None:
    #         print(v, value(v))


def print_reports(m):

    print('')
    for unit_k in [m.fs.boiler, m.fs.reheater[1],
                   m.fs.reheater[2],
                   m.fs.bfp, m.fs.bfpt,
                   m.fs.booster,
                   m.fs.condenser_mix,
                   m.fs.charge.ess_vhp_split,
                   m.fs.charge.solar_salt_disjunct.hxc,
                   m.fs.charge.hitec_salt_disjunct.hxc,
                   m.fs.charge.thermal_oil_disjunct.hxc]:
        unit_k.display()

    for k in pyo.RangeSet(11):
        m.fs.turbine[k].report()
    for k in pyo.RangeSet(11):
        m.fs.turbine[k].display()
    for j in pyo.RangeSet(9):
        m.fs.fwh[j].report()
    for j in m.set_fwh_mixer:
        m.fs.fwh_mixer[j].display()


def model_analysis(m, solver, heat_duty=None, source=None):
    """Unfix variables for analysis. This section is deactived for the
    simulation of square model
    """

    # Fix variables in the flowsheet
    m.fs.plant_power_out.fix(400)
    m.fs.boiler.outlet.pressure.fix(m.main_steam_pressure)
    m.fs.discharge.hxd.heat_duty.fix(heat_duty*1e6)  # in W

    # Unfix variables fixed in model input and during initialization
    m.fs.boiler.inlet.flow_mol.unfix()  # mol/s
    # m.fs.boiler.inlet.flow_mol.fix(m.main_flow)  # mol/s

    m.fs.discharge.es_split.split_fraction[0, "to_hxd"].unfix()

    m.fs.discharge.es_split.inlet.flow_mol.unfix()   # kg/s
    m.fs.discharge.es_split.inlet.enth_mol.unfix()  # K
    m.fs.discharge.es_split.inlet.pressure.unfix()  # Pa

    m.fs.discharge.hxd.inlet_1.flow_mass.unfix()
    m.fs.discharge.hxd.area.unfix()

    # Objective function: total costs
    m.obj = Objective(
        expr=(
            m.fs.discharge.capital_cost
            + m.fs.discharge.operating_cost
        )
    )

    print('DOF before solution = ', degrees_of_freedom(m))

    # Solve the design optimization modelfwh4
    # results = run_nlps(m,
    #                    solver=solver,
    #                    source=source)

    results = run_gdp(m)

    print_results(m, results)
    # print_reports(m)

    return m


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    m_usc = usc.build_plant_model()
    usc.initialize(m_usc)

    m, solver = main(m_usc)

    # source = ["condpump", "booster", "bfp", "fwh9"]

    m = model_analysis(m, solver, heat_duty=148.5, source="fwh9")
    # results = solver.solve(
    #     m,
    #     tee=True,
    #     symbolic_solver_labels=True,
    #     options={
    #         "linear_solver": "ma27",
    #         "max_iter": 150,
    #         # "bound_push": 1e-12,
    #         # "mu_init": 1e-8
    #     }
    # )
    # print('Charge heat duty (MW):', k)
