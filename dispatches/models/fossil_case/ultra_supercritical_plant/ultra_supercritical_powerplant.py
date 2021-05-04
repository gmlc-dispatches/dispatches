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

"""
This is a simple model for an ultrasupercritical coal-fired
power plant based on a flowsheet presented in 1999 USDOE Report #DOE/FE-0400

This model uses some of the simpler unit models from the
power generation unit model library.

Some of the parameters in the model such as feed water heater areas,
overall heat transfer coefficient, turbine efficiencies at multiple stages
have all been estimated for a total power out of 437 MW.

Additional main assumptions are as follows:
    (1) The flowsheet and main steam conditions, i. e. pressure & temperature
        are adopted from the aforementioned DOE report
    (2) Heater unit models are used to model main steam boiler,
        reheater, and condenser.
    (3) Multi-stage turbines are modeled as 
        multiple lumped single stage turbines

updated (02/24/2021)
"""

__author__ = "Naresh Susarla"

import os

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
from pyomo.common.fileutils import this_file_dir

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import get_solver, copy_port_values as _set_port
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (
    HeatExchanger,
    MomentumMixingType,
    Heater,
)
from idaes.power_generation.unit_models.helm import (
    HelmMixer,
    HelmIsentropicCompressor,
    HelmTurbineStage,
    HelmSplitter
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.misc import svg_tag

# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.generic_models.properties import iapws95


def declare_unit_model():
    """Create flowsheet and add unit models.
    """
    ###########################################################################
    #  Flowsheet and Property Package                                         #
    ###########################################################################
    m = pyo.ConcreteModel(name="Steam Cycle Model")
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.prop_water = iapws95.Iapws95ParameterBlock()

    ###########################################################################
    #   Turbine declarations                                   #
    ###########################################################################
    # A total of 11 single stage turbines are used to model
    # different multistage turbines
    m.fs.turbine = HelmTurbineStage(
        pyo.RangeSet(11),
        default={
            "property_package": m.fs.prop_water,
        }
    )

    #########################################################################
    #  Turbine outlet splitters                                  #
    #########################################################################
    # The default number of outlets for a splitter is 2. This can be changed
    # using the "num_outlets" argument.
    # In the USC flowsheet turbine_splitter[6] has 3 outlets. This is realized
    # by using the 'initialize' argument as shown below.
    m.fs.turbine_splitter = HelmSplitter(
        pyo.RangeSet(10),
        default = {
            "property_package": m.fs.prop_water
            },
        initialize={
            6:{
                "property_package": m.fs.prop_water,
                "num_outlets": 3
                },
        }
    )

    ###########################################################################
    #  Boiler section declarations:                                #
    ###########################################################################
    # Boiler section is set up using three heater blocks, as following:
    # 1) For the main steam the heater block is named 'boiler'
    # 2) For the 1st reheated steam the heater block is named 'reheater_1'
    # 3) For the 2nd reheated steam the heater block is named 'reheater_2'
    m.fs.boiler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    m.fs.reheater_1 = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    m.fs.reheater_2 = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )

    ###########################################################################
    #  Add Condenser Mixer, Condenser, and Condensate pump                    #
    ###########################################################################
    # condenser mix
    m.fs.condenser_mix = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["main", "bfpt", "drain", "makeup"],
            "property_package": m.fs.prop_water,
        }
    )

    # Condenser is set up as a heater block
    m.fs.condenser = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": False
        }
    )

    # condensate pump
    m.fs.cond_pump = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    ###########################################################################
    #  Feedwater heater declaration                                     #
    ###########################################################################
    # Feed water heaters (FWHs) are declared as 0D heat exchangers
    # Shell side (side 1) is for steam condensing
    # Tube side (side 2) is for feed water heating 

    # Declaring indexed units for feed water heater mixers
    # The indices reflect the connection to corresponding feed water heaters
    # e. g. the outlet of fwh_mixer[1] is connected to fwh[1]
    # Note that there are no mixers before FWHs 5 and 9
    m.mixer_list = [1, 2, 3, 4, 6, 7, 8]
    m.fs.fwh_mixer = HelmMixer(
        m.mixer_list,
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )


    # Declaring indexed units for feed water heaters
    m.fs.fwh = HeatExchanger(
        pyo.RangeSet(9),
        default={
            "delta_temperature_callback": delta_temperature_underwood_callback,
            "shell": {
                "property_package": m.fs.prop_water,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "has_pressure_change": True,
            },
            "tube": {
                "property_package": m.fs.prop_water,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "has_pressure_change": True,
            },
        }
    )

    ###########################################################################
    #  Add deaerator                              #
    ###########################################################################
    m.fs.deaerator = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "inlet_list": ["steam", "drain", "feedwater"],
            "property_package": m.fs.prop_water,
        }
    )

    ###########################################################################
    #  Add auxiliary turbine, booster pump, and boiler feed pump (BFP)        #
    ###########################################################################
    m.fs.booster = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    m.fs.bfp = HelmIsentropicCompressor(
        default={
            "property_package": m.fs.prop_water,
        }
    )
    m.fs.bfpt = HelmTurbineStage(
        default={
            "property_package": m.fs.prop_water,
        }
    )

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _make_constraints(m)
    _create_arcs(m)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)
    return m

def _make_constraints(m):
    # Define all model constraints except those included with the unit models
    
    #########   Boiler section   #########

    # Following the reference DOE flowsheet, the outlet temperature for the
    # boiler unit, i. e. boiler, reheater_1, & reheater_2 is fixed to 866.15 K
    # Outlet temperature of boiler is set to 866.15 K
    @m.fs.boiler.Constraint(m.fs.time)
    def boiler_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature ==
                866.15 * pyunits.K)

    # Outlet temperature of reheater_1 is set to 866.15 K
    @m.fs.reheater_1.Constraint(m.fs.time)
    def reheater1_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature ==
                866.15 * pyunits.K)

    # Outlet temperature of reheater_2 is set to 866.15 K
    @m.fs.reheater_2.Constraint(m.fs.time)
    def reheater2_temperature_constraint(b, t):
        return (b.control_volume.properties_out[t].temperature ==
                866.15 * pyunits.K)

    #########   Condenser section   #########
    
    # The inlet 'main' refers to the main steam coming from the turbine train
    # Inlet 'bfpt' refers to the steam coming from the bolier feed pump turbine
    # Inlet 'drain' refers to the condensed steam from the feed water heater 1
    # Inlet 'makeup' refers to the make up water
    # The outlet pressure of condenser mixer is equal to the minimum pressure
    # Since the turbine (#11) outlet (or, mixer inlet 'main') pressure
    # has the minimum pressure, the following constraint sets the outlet
    # pressure of the condenser mixer to the pressure of the inlet 'main'
    @m.fs.condenser_mix.Constraint(m.fs.time)
    def mixer_pressure_constraint(b, t):
        return b.main_state[t].pressure == b.mixed_state[t].pressure

    # The outlet of condenser is assumed to be a saturated liquid
    # The following condensate enthalpy at the outlet of condeser equal to
    # that of a saturated liquid at that pressure
    @m.fs.condenser.Constraint(m.fs.time)
    def cond_vaporfrac_constraint(b, t):
        return (
            b.control_volume.properties_out[t].enth_mol
            == b.control_volume.properties_out[t].enth_mol_sat_phase['Liq']
        )

    #########   Feed Water Heater section   #########

    # The condensing steam is assumed to leave the FWH as saturated liquid
    # Thus, each FWH is accompanied by 3 constraints, 2 for pressure drop
    # and 1 for the enthalpy.

    # The outlet pressure of fwh mixer is equal to the minimum pressure
    # Since the pressure of mixer inlet 'steam' has the minimum pressure,
    # the following constraint set the outlet pressure of fwh mixers
    # to be same as the pressure of the inlet 'steam'
    def fwhmixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    for p in m.mixer_list:
        b = m.fs.fwh_mixer[p]
        setattr(b,
                "mixer_press_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwhmixer_pressure_constraint))

    # Side 1 outlet of fwh is assumed to be a saturated liquid
    # The following constraint sets the side 1 outlet enthalpy to be 
    # same as that of saturated liquid
    def fwh_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )
    for i in pyo.RangeSet(9):
        b = m.fs.fwh[i]
        setattr(b,
                "fwh_vfrac_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwh_vaporfrac_constraint))

    # Pressure drop on both sides are accounted for by setting the respective
    # outlet pressure based on the following assumptions:
    #     (1) Feed water side (side 2): A constant 4% pressure drop is assumed
    #           on the feedwater side for all FWHs. For this,
    #           the outlet pressure is set to 0.96 times the inlet pressure,
    #           on the feed water side for all FWHs
    #     (2) Steam condensing side (side 1): Going from high pressure to
    #           low pressure FWHs, the outlet pressure of
    #           the condensed steam in assumed to be 10% more than that
    #           of the pressure of steam extracted for the immediately
    #           next lower pressure feedwater heater.
    #           e.g. the outlet condensate pressure of FWH 'n'
    #           = 1.1 * pressure of steam extracted for FWH 'n-1'
    #           In case of FWH1 the FWH 'n-1' is used for Condenser,
    #           and in case of FWH6, FWH 'n-1' is for Deaerator. Here,
    #           the steam pressure for FWH 'n-1' is known because the
    #           pressure ratios for turbines are fixed.

    # Side 2 pressure drop constraint
    # Setting a 4% pressure drop on the feedwater side (P_out = 0.96 * P_in)
    def fwh_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )
    for i in pyo.RangeSet(9):
        b = m.fs.fwh[i]
        setattr(b,
                "fwh_s2_delp_constraint",
                pyo.Constraint(m.fs.config.time,
                               rule=fwh_s2pdrop_constraint))

    # Side 1 pressure drop constraint
    # Setting the outlet pressure as described above. For this, the
    # relevant turbine stage pressure ratios are used (pressure_ratio_dict)
    # The pressure drop across the reheaters 1 and 2 are also accounted
    # in case of fwh[9] and fwh[7], respectively
    # For this, pressure_diffr_dict is defined

    # 0.204 is the pressure ratio for turbine #11 (see set_inputs)
    # 0.476 is the pressure ratio for turbine #10 (see set_inputs)
    # 0.572 is the pressure ratio for turbine #9 (see set_inputs)
    # 0.389 is the pressure ratio for turbine #8 (see set_inputs)
    # 0.514 is the pressure ratio for turbine #7 (see set_inputs)
    # 0.523 is the pressure ratio for turbine #5 (see set_inputs)
    # 0.609 is the pressure ratio for turbine #4 (see set_inputs)
    # 0.498 is the pressure ratio for turbine #3 (see set_inputs)
    # 0.774 is the pressure ratio for turbine #2 (see set_inputs)
    pressure_ratio_dict = {1: 0.204,
                           2: 0.476,
                           3: 0.572,
                           4: 0.389,
                           5: 0.514,
                           6: 0.523,
                           7: 0.609,
                           8: 0.498,
                           9: 0.774}

    # 742845 Pa is the pressure drop across reheater_1
    # 210952 Pa is the pressure drop across reheater_2
    pressure_diffr_dict = {1: 0,
                           2: 0,
                           3: 0,
                           4: 0,
                           5: 0,
                           6: 210952,
                           7: 0,
                           8: 742845,
                           9: 0}
    
    def fwh_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * b.turb_press_ratio *
            (b.side_1.properties_in[t].pressure - b.reheater_press_diff)
        )

    for i in pyo.RangeSet(9):
        b = m.fs.fwh[i]
        b.turb_press_ratio = pyo.Param(initialize = pressure_ratio_dict[i],
                                       units = pyunits.Pa/pyunits.Pa)
        b.reheater_press_diff = pyo.Param(initialize = pressure_diffr_dict[i],
                                          units = pyunits.Pa)
        setattr(b, "s1_delp_constraint",
                pyo.Constraint(m.fs.config.time, rule=fwh_s1pdrop_constraint))

    # The outlet pressure of deaerator is equal to the minimum pressure
    # Since the pressure of deaerator inlet 'feedwater' has
    # the minimum pressure, the following constraint sets the outlet pressure
    # of deaerator to be same as the pressure of the inlet 'feedwater'
    @m.fs.deaerator.Constraint(m.fs.time)
    def damixer_pressure_constraint(b, t):
        return b.feedwater_state[t].pressure == b.mixed_state[t].pressure

    # The following constraint sets the outlet pressure of steam extracted
    # for boiler feed water turbine to be same as that of condenser
    @m.fs.Constraint(m.fs.time)
    def constraint_out_pressure(b, t):
        return (
            b.bfpt.control_volume.properties_out[t].pressure
            == b.condenser_mix.main_state[t].pressure
        )

    # The following constraint demands that the work done by the
    # boiler feed water pump is same as that of boiler feed water turbine
    # Essentially, this says that boiler feed water turbine produces just
    # enough power to meet the demand of boiler feed water pump
    @m.fs.Constraint(m.fs.time)
    def constraint_bfp_power(b, t):
        return (
            b.booster.control_volume.work[t] + b.bfp.control_volume.work[t]
            + b.bfpt.control_volume.work[t]
            == 0
        )


def _create_arcs(m):

    # boiler to turb
    m.fs.boiler_to_turb1 = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.turbine[1].inlet
    )

    # turbine1 splitter
    m.fs.turb1_to_t1split = Arc(
        source=m.fs.turbine[1].outlet,
        destination=m.fs.turbine_splitter[1].inlet
    )
    m.fs.t1split_to_turb2 = Arc(
        source=m.fs.turbine_splitter[1].outlet_1,
        destination=m.fs.turbine[2].inlet
    )
    m.fs.t1split_to_fwh10 = Arc(
        source=m.fs.turbine_splitter[1].outlet_2,
        destination=m.fs.fwh[9].inlet_1
    )

    # turbine2 splitter
    m.fs.turb2_to_t2split = Arc(
        source=m.fs.turbine[2].outlet,
        destination=m.fs.turbine_splitter[2].inlet
    )
    m.fs.t2split_to_rh1 = Arc(
        source=m.fs.turbine_splitter[2].outlet_1,
        destination=m.fs.reheater_1.inlet
    )
    m.fs.t2split_to_fwh9mix = Arc(
        source=m.fs.turbine_splitter[2].outlet_2,
        destination=m.fs.fwh_mixer[8].steam
    )

    # reheater_1 to turbine_3
    m.fs.rh1_to_turb3 = Arc(
        source=m.fs.reheater_1.outlet, destination=m.fs.turbine[3].inlet
    )

    # turbine3 splitter
    m.fs.turb3_to_t3split = Arc(
        source=m.fs.turbine[3].outlet,
        destination=m.fs.turbine_splitter[3].inlet
    )
    m.fs.t3split_to_turb4 = Arc(
        source=m.fs.turbine_splitter[3].outlet_1,
        destination=m.fs.turbine[4].inlet
    )
    m.fs.t3split_to_fwh8mix = Arc(
        source=m.fs.turbine_splitter[3].outlet_2,
        destination=m.fs.fwh_mixer[7].steam
    )

    # turbine4 splitter
    m.fs.turb4_to_t4split = Arc(
        source=m.fs.turbine[4].outlet,
        destination=m.fs.turbine_splitter[4].inlet
    )
    m.fs.t4split_to_rh2 = Arc(
        source=m.fs.turbine_splitter[4].outlet_1,
        destination=m.fs.reheater_2.inlet
    )
    m.fs.t4split_to_fwh7mix = Arc(
        source=m.fs.turbine_splitter[4].outlet_2,
        destination=m.fs.fwh_mixer[6].steam
    )

    # reheater_2 to turbine_5
    m.fs.rh2_to_turb5 = Arc(
        source=m.fs.reheater_2.outlet, destination=m.fs.turbine[5].inlet
    )

    # turbine5 splitter
    m.fs.turb5_to_t5split = Arc(
        source=m.fs.turbine[5].outlet,
        destination=m.fs.turbine_splitter[5].inlet
    )
    m.fs.t5split_to_turb6 = Arc(
        source=m.fs.turbine_splitter[5].outlet_1,
        destination=m.fs.turbine[6].inlet
    )
    m.fs.t5split_to_fwh6da = Arc(
        source=m.fs.turbine_splitter[5].outlet_2,
        destination=m.fs.deaerator.steam
    )

    # turbine6 splitter
    m.fs.turb6_to_t6split = Arc(
        source=m.fs.turbine[6].outlet,
        destination=m.fs.turbine_splitter[6].inlet
    )
    m.fs.t6split_to_turb7 = Arc(
        source=m.fs.turbine_splitter[6].outlet_1,
        destination=m.fs.turbine[7].inlet
    )
    m.fs.t6split_to_fwh5 = Arc(
        source=m.fs.turbine_splitter[6].outlet_2,
        destination=m.fs.fwh[5].inlet_1
    )
    m.fs.t6split_to_bfpt = Arc(
        source=m.fs.turbine_splitter[6].outlet_3,
        destination=m.fs.bfpt.inlet
    )

    # turbine7 splitter
    m.fs.turb7_to_t7split = Arc(
        source=m.fs.turbine[7].outlet,
        destination=m.fs.turbine_splitter[7].inlet
    )
    m.fs.t7split_to_turb8 = Arc(
        source=m.fs.turbine_splitter[7].outlet_1,
        destination=m.fs.turbine[8].inlet
    )
    m.fs.t7split_to_fwh4mix = Arc(
        source=m.fs.turbine_splitter[7].outlet_2,
        destination=m.fs.fwh_mixer[4].steam
    )

    # turbine8 splitter
    m.fs.turb8_to_t8split = Arc(
        source=m.fs.turbine[8].outlet,
        destination=m.fs.turbine_splitter[8].inlet
    )
    m.fs.t8split_to_turb9 = Arc(
        source=m.fs.turbine_splitter[8].outlet_1,
        destination=m.fs.turbine[9].inlet
    )
    m.fs.t8split_to_fwh3mix = Arc(
        source=m.fs.turbine_splitter[8].outlet_2,
        destination=m.fs.fwh_mixer[3].steam
    )

    # turbine9 splitter
    m.fs.turb9_to_t9split = Arc(
        source=m.fs.turbine[9].outlet,
        destination=m.fs.turbine_splitter[9].inlet
    )
    m.fs.t9split_to_turb10 = Arc(
        source=m.fs.turbine_splitter[9].outlet_1,
        destination=m.fs.turbine[10].inlet
    )
    m.fs.t9split_to_fwh2mix = Arc(
        source=m.fs.turbine_splitter[9].outlet_2,
        destination=m.fs.fwh_mixer[2].steam
    )

    # turbine10 splitter
    m.fs.turb10_to_t10split = Arc(
        source=m.fs.turbine[10].outlet,
        destination=m.fs.turbine_splitter[10].inlet
    )
    m.fs.t10split_to_turb11 = Arc(
        source=m.fs.turbine_splitter[10].outlet_1,
        destination=m.fs.turbine[11].inlet
    )
    m.fs.t10split_to_fwh1mix = Arc(
        source=m.fs.turbine_splitter[10].outlet_2,
        destination=m.fs.fwh_mixer[1].steam
    )

    # condenser mixer to condensate pump
    m.fs.turb11_to_cmix = Arc(
        source=m.fs.turbine[11].outlet,
        destination=m.fs.condenser_mix.main
    )
    m.fs.drain_to_cmix = Arc(
        source=m.fs.fwh[1].outlet_1,
        destination=m.fs.condenser_mix.drain
    )
    m.fs.bfpt_to_cmix = Arc(
        source=m.fs.bfpt.outlet,
        destination=m.fs.condenser_mix.bfpt
    )
    m.fs.cmix_to_cond = Arc(
        source=m.fs.condenser_mix.outlet,
        destination=m.fs.condenser.inlet
    )
    m.fs.cond_to_cpump = Arc(
        source=m.fs.condenser.outlet, destination=m.fs.cond_pump.inlet
    )

    # fwh1
    m.fs.pump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet, destination=m.fs.fwh[1].inlet_2
    )
    m.fs.fwh2_to_fwh1mix = Arc(
        source=m.fs.fwh[2].outlet_1, destination=m.fs.fwh_mixer[1].drain
    )
    m.fs.mix1_to_fwh1 = Arc(
        source=m.fs.fwh_mixer[1].outlet, destination=m.fs.fwh[1].inlet_1
    )

    # fwh2
    m.fs.fwh3_to_fwh2mix = Arc(
        source=m.fs.fwh[3].outlet_1, destination=m.fs.fwh_mixer[2].drain
    )
    m.fs.mix2_to_fwh2 = Arc(
        source=m.fs.fwh_mixer[2].outlet, destination=m.fs.fwh[2].inlet_1
    )
    m.fs.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh[1].outlet_2, destination=m.fs.fwh[2].inlet_2
    )

    # fwh3
    m.fs.fwh4_to_fwh3mix = Arc(
        source=m.fs.fwh[4].outlet_1, destination=m.fs.fwh_mixer[3].drain
    )
    m.fs.mix3_to_fwh3 = Arc(
        source=m.fs.fwh_mixer[3].outlet, destination=m.fs.fwh[3].inlet_1
    )
    m.fs.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh[2].outlet_2, destination=m.fs.fwh[3].inlet_2
    )

    # fwh4
    m.fs.fwh5_to_fwh4mix = Arc(
        source=m.fs.fwh[5].outlet_1, destination=m.fs.fwh_mixer[4].drain
    )
    m.fs.mix4_to_fwh4 = Arc(
        source=m.fs.fwh_mixer[4].outlet, destination=m.fs.fwh[4].inlet_1
    )
    m.fs.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh[3].outlet_2, destination=m.fs.fwh[4].inlet_2
    )

    # fwh5
    m.fs.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh[4].outlet_2, destination=m.fs.fwh[5].inlet_2
    )

    # Deaerator
    m.fs.fwh5_to_fwh6da = Arc(
        source=m.fs.fwh[5].outlet_2, destination=m.fs.deaerator.feedwater
    )
    m.fs.fwh7_to_fwh6da = Arc(
        source=m.fs.fwh[6].outlet_1, destination=m.fs.deaerator.drain
    )

    # Booster Pump
    m.fs.fwh6da_to_booster = Arc(
        source=m.fs.deaerator.outlet, destination=m.fs.booster.inlet
    )

    # fwh7
    m.fs.fwh8_to_fwh7mix = Arc(
        source=m.fs.fwh[7].outlet_1, destination=m.fs.fwh_mixer[6].drain
    )
    m.fs.mix7_to_fwh7 = Arc(
        source=m.fs.fwh_mixer[6].outlet, destination=m.fs.fwh[6].inlet_1
    )
    m.fs.booster_to_fwh7 = Arc(
        source=m.fs.booster.outlet, destination=m.fs.fwh[6].inlet_2
    )

    # fwh8
    m.fs.fwh9_to_fwh8mix = Arc(
        source=m.fs.fwh[8].outlet_1, destination=m.fs.fwh_mixer[7].drain
    )
    m.fs.mix8_to_fwh8 = Arc(
        source=m.fs.fwh_mixer[7].outlet, destination=m.fs.fwh[7].inlet_1
    )
    m.fs.fwh7_to_fwh8 = Arc(
        source=m.fs.fwh[6].outlet_2, destination=m.fs.fwh[7].inlet_2
    )

    # BFW Pump
    m.fs.fwh8_to_bfp = Arc(
        source=m.fs.fwh[7].outlet_2, destination=m.fs.bfp.inlet
    )

    # fwh9
    m.fs.fwh10_to_fwh9mix = Arc(
        source=m.fs.fwh[9].outlet_1, destination=m.fs.fwh_mixer[8].drain
    )
    m.fs.mix9_to_fwh9 = Arc(
        source=m.fs.fwh_mixer[8].outlet, destination=m.fs.fwh[8].inlet_1
    )
    m.fs.bfp_to_fwh9 = Arc(
        source=m.fs.bfp.outlet, destination=m.fs.fwh[8].inlet_2
    )

    # fwh10
    m.fs.fwh9_to_fwh10 = Arc(
        source=m.fs.fwh[8].outlet_2, destination=m.fs.fwh[9].inlet_2
    )

    # boiler
    m.fs.fwh10_to_boiler = Arc(
        source=m.fs.fwh[9].outlet_2, destination=m.fs.boiler.inlet
    )


def set_model_input(m):

    # Model inputs / fixed variable or parameter values
    # assumed in this block, unless otherwise stated explicitly,
    # are either assumed or estimated for a total power out of 437 MW

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    ###########################################################################
    #  Turbine input                                                          #
    ###########################################################################
    #  Turbine inlet conditions
    main_steam_pressure = 31125980  # Pa
    m.fs.boiler.inlet.flow_mol.fix(17854)  # mol/s
    m.fs.boiler.outlet.pressure.fix(main_steam_pressure)

    # Reheater section pressure drop estimated
    # for a total power out of 437 MW
    m.fs.reheater_1.deltaP.fix(-742845)  # Pa
    m.fs.reheater_2.deltaP.fix(-210952)  # Pa

    # The efficiency and pressure ratios of all turbines were estimated
    # for a total power out of 437 MW
    m.fs.turbine[1].ratioP.fix(0.388)
    m.fs.turbine[1].efficiency_isentropic.fix(0.94)

    m.fs.turbine[2].ratioP.fix(0.774)
    m.fs.turbine[2].efficiency_isentropic.fix(0.94)

    m.fs.turbine[3].ratioP.fix(0.498)
    m.fs.turbine[3].efficiency_isentropic.fix(0.94)

    m.fs.turbine[4].ratioP.fix(0.609)
    m.fs.turbine[4].efficiency_isentropic.fix(0.94)

    m.fs.turbine[5].ratioP.fix(0.523)
    m.fs.turbine[5].efficiency_isentropic.fix(0.88)

    m.fs.turbine[6].ratioP.fix(0.495)
    m.fs.turbine[6].efficiency_isentropic.fix(0.88)

    m.fs.turbine[7].ratioP.fix(0.514)
    m.fs.turbine[7].efficiency_isentropic.fix(0.78)

    m.fs.turbine[8].ratioP.fix(0.389)
    m.fs.turbine[8].efficiency_isentropic.fix(0.78)

    m.fs.turbine[9].ratioP.fix(0.572)
    m.fs.turbine[9].efficiency_isentropic.fix(0.78)

    m.fs.turbine[10].ratioP.fix(0.476)
    m.fs.turbine[10].efficiency_isentropic.fix(0.78)

    m.fs.turbine[11].ratioP.fix(0.204)
    m.fs.turbine[11].efficiency_isentropic.fix(0.78)

    ###########################################################################
    #  Condenser section                                         #
    ###########################################################################
    m.fs.cond_pump.efficiency_isentropic.fix(0.80)
    m.fs.cond_pump.deltaP.fix(2313881)

    # Make up stream to condenser
    m.fs.condenser_mix.makeup.flow_mol.value = -9.0E-12  # mol/s
    m.fs.condenser_mix.makeup.pressure.fix(103421.4)  # Pa
    m.fs.condenser_mix.makeup.enth_mol.fix(1131.69204)  # J/mol

    ###########################################################################
    #  Low pressure FWH section inputs                                        #
    ###########################################################################
    # fwh1
    m.fs.fwh[1].area.fix(250)
    m.fs.fwh[1].overall_heat_transfer_coefficient.fix(3000)
    # fwh2
    m.fs.fwh[2].area.fix(195)
    m.fs.fwh[2].overall_heat_transfer_coefficient.fix(3000)
    # fwh3
    m.fs.fwh[3].area.fix(164)
    m.fs.fwh[3].overall_heat_transfer_coefficient.fix(3000)
    # fwh4
    m.fs.fwh[4].area.fix(208)
    m.fs.fwh[4].overall_heat_transfer_coefficient.fix(3000)
    # fwh5
    m.fs.fwh[5].area.fix(152)
    m.fs.fwh[5].overall_heat_transfer_coefficient.fix(3000)

    #########################################################################
    #  Deaerator and boiler feed pump (BFP) Input                           #
    #########################################################################
    # Unlike the feedwater heaters the steam extraction flow to the deaerator
    # is not constrained by the saturated liquid constraint. Thus, the flow
    # to the deaerator is assumed to be fixed in this model.
    m.fs.turbine_splitter[5].split_fraction[:, "outlet_2"].fix(0.017885)

    m.fs.booster.efficiency_isentropic.fix(0.80)
    m.fs.booster.deltaP.fix(5715067)
    # BFW Pump pressure is assumed based on referece report
    m.fs.bfp.outlet.pressure[:].fix(main_steam_pressure * 1.1231)  # Pa
    m.fs.bfp.efficiency_isentropic.fix(0.80)

    m.fs.bfpt.efficiency_isentropic.fix(0.80)
    ###########################################################################
    #  High pressure feedwater heater                                         #
    ###########################################################################
    # fwh6
    m.fs.fwh[6].area.fix(207)  # 300
    m.fs.fwh[6].overall_heat_transfer_coefficient.fix(3000)
    # fwh7
    m.fs.fwh[7].area.fix(202)  # 202
    m.fs.fwh[7].overall_heat_transfer_coefficient.fix(3000)
    # fwh8
    m.fs.fwh[8].area.fix(715)  # 715
    m.fs.fwh[8].overall_heat_transfer_coefficient.fix(3000)
    # fwh9
    m.fs.fwh[9].area.fix(175)  # 275
    m.fs.fwh[9].overall_heat_transfer_coefficient.fix(3000)


def set_scaling_factors(m):
    # scaling factors in the flowsheet

    for i in pyo.RangeSet(9):
        b = m.fs.fwh[i]
        iscale.set_scaling_factor(b.area, 1e-2)
        iscale.set_scaling_factor(b.overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(b.shell.heat, 1e-6)
        iscale.set_scaling_factor(b.tube.heat, 1e-6)

    for j in pyo.RangeSet(11):
        b = m.fs.turbine[j]
        iscale.set_scaling_factor(b.control_volume.work, 1e-6)
    
    iscale.set_scaling_factor(m.fs.boiler.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reheater_1.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.reheater_2.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)
    iscale.set_scaling_factor(m.fs.cond_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.booster.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.bfp.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.bfpt.control_volume.work, 1e-6)
    
    # return m

def initialize(m, fileinput=None, outlvl=idaeslog.NOTSET,
               solver=None, optarg={}):

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

    iscale.calculate_scaling_factors(m)

    # # initializing the boiler
    m.fs.boiler.inlet.pressure.fix(32216913)
    m.fs.boiler.inlet.enth_mol.fix(23737)
    m.fs.boiler.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.boiler.inlet.pressure.unfix()
    m.fs.boiler.inlet.enth_mol.unfix()

    # initialization routine for the turbine train

    # Deactivating constraints that fix enthalpy at FWH outlet
    # This lets us initialize the model using the fixed split_fractions
    # for steam extractions for all the feed water heaters except deaerator
    # These split fractions will be unfixed later and the constraints will
    # be activated
    m.fs.turbine_splitter[1].split_fraction[:, "outlet_2"].fix(0.073444)
    m.fs.turbine_splitter[2].split_fraction[:, "outlet_2"].fix(0.140752)
    m.fs.turbine_splitter[3].split_fraction[:, "outlet_2"].fix(0.032816)
    m.fs.turbine_splitter[4].split_fraction[:, "outlet_2"].fix(0.012425)
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_2"].fix(0.081155)
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_3"].fix(0.091274)
    m.fs.turbine_splitter[7].split_fraction[:, "outlet_2"].fix(0.036058)
    m.fs.turbine_splitter[8].split_fraction[:, "outlet_2"].fix(0.026517)
    m.fs.turbine_splitter[9].split_fraction[:, "outlet_2"].fix(0.029888)
    m.fs.turbine_splitter[10].split_fraction[:, "outlet_2"].fix(0.003007)

    m.fs.constraint_bfp_power.deactivate()
    m.fs.constraint_out_pressure.deactivate()
    for i in pyo.RangeSet(9):
        m.fs.fwh[i].fwh_vfrac_constraint.deactivate()

    # solving the turbine, splitter, and reheaters
    _set_port(m.fs.turbine[1].inlet,  m.fs.boiler.outlet)
    m.fs.turbine[1].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[1].inlet,  m.fs.turbine[1].outlet)
    m.fs.turbine_splitter[1].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[2].inlet,  m.fs.turbine_splitter[1].outlet_1)
    m.fs.turbine[2].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[2].inlet,  m.fs.turbine[2].outlet)
    m.fs.turbine_splitter[2].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.reheater_1.inlet,  m.fs.turbine_splitter[2].outlet_1)
    m.fs.reheater_1.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[3].inlet, m.fs.reheater_1.outlet)
    m.fs.turbine[3].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[3].inlet,  m.fs.turbine[3].outlet)
    m.fs.turbine_splitter[3].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[4].inlet,  m.fs.turbine_splitter[3].outlet_1)
    m.fs.turbine[4].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[4].inlet,  m.fs.turbine[4].outlet)
    m.fs.turbine_splitter[4].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.reheater_2.inlet,  m.fs.turbine_splitter[4].outlet_1)
    m.fs.reheater_2.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[5].inlet,  m.fs.reheater_2.outlet)
    m.fs.turbine[5].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[5].inlet,  m.fs.turbine[5].outlet)
    m.fs.turbine_splitter[5].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[6].inlet,  m.fs.turbine_splitter[5].outlet_1)
    m.fs.turbine[6].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[6].inlet,  m.fs.turbine[6].outlet)
    m.fs.turbine_splitter[6].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[7].inlet,  m.fs.turbine_splitter[6].outlet_1)
    m.fs.turbine[7].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[7].inlet,  m.fs.turbine[7].outlet)
    m.fs.turbine_splitter[7].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[8].inlet,  m.fs.turbine_splitter[7].outlet_1)
    m.fs.turbine[8].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[8].inlet,  m.fs.turbine[8].outlet)
    m.fs.turbine_splitter[8].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[9].inlet,  m.fs.turbine_splitter[8].outlet_1)
    m.fs.turbine[9].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[9].inlet,  m.fs.turbine[9].outlet)
    m.fs.turbine_splitter[9].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[10].inlet,  m.fs.turbine_splitter[9].outlet_1)
    m.fs.turbine[10].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_splitter[10].inlet,  m.fs.turbine[10].outlet)
    m.fs.turbine_splitter[10].initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine[11].inlet,  m.fs.turbine_splitter[10].outlet_1)
    m.fs.turbine[11].initialize(outlvl=outlvl, optarg=solver.options)

    # initialize the boiler feed pump turbine.
    _set_port(m.fs.bfpt.inlet,  m.fs.turbine_splitter[6].outlet_3)
    m.fs.bfpt.outlet.pressure.fix(6896)
    m.fs.bfpt.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.bfpt.outlet.pressure.unfix()

    ###########################################################################
    #  Condenser                                                #
    ###########################################################################
    _set_port(m.fs.condenser_mix.bfpt, m.fs.bfpt.outlet)
    _set_port(m.fs.condenser_mix.main, m.fs.turbine[11].outlet)
    m.fs.condenser_mix.drain.flow_mol.fix(2102)
    m.fs.condenser_mix.drain.pressure.fix(7586)
    m.fs.condenser_mix.drain.enth_mol.fix(3056)
    m.fs.condenser_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.condenser_mix.drain.unfix()

    _set_port(m.fs.condenser.inlet, m.fs.condenser_mix.outlet)
    m.fs.condenser.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.cond_pump.inlet, m.fs.condenser.outlet)
    m.fs.cond_pump.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  Low pressure FWH section                                               #
    ###########################################################################

    # fwh1
    _set_port(m.fs.fwh_mixer[1].steam,  m.fs.turbine_splitter[10].outlet_2)
    m.fs.fwh_mixer[1].drain.flow_mol.fix(2072)
    m.fs.fwh_mixer[1].drain.pressure.fix(37187)
    m.fs.fwh_mixer[1].drain.enth_mol.fix(5590)
    m.fs.fwh_mixer[1].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[1].drain.unfix()

    _set_port(m.fs.fwh[1].inlet_1, m.fs.fwh_mixer[1].outlet)
    _set_port(m.fs.fwh[1].inlet_2, m.fs.cond_pump.outlet)
    m.fs.fwh[1].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh2
    _set_port(m.fs.fwh_mixer[2].steam,  m.fs.turbine_splitter[9].outlet_2)
    m.fs.fwh_mixer[2].drain.flow_mol.fix(1762)
    m.fs.fwh_mixer[2].drain.pressure.fix(78124)
    m.fs.fwh_mixer[2].drain.enth_mol.fix(7009)
    m.fs.fwh_mixer[2].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[2].drain.unfix()

    _set_port(m.fs.fwh[2].inlet_1, m.fs.fwh_mixer[2].outlet)
    _set_port(m.fs.fwh[2].inlet_2, m.fs.fwh[1].outlet_2)
    m.fs.fwh[2].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh3
    _set_port(m.fs.fwh_mixer[3].steam,  m.fs.turbine_splitter[8].outlet_2)
    m.fs.fwh_mixer[3].drain.flow_mol.fix(1480)
    m.fs.fwh_mixer[3].drain.pressure.fix(136580)
    m.fs.fwh_mixer[3].drain.enth_mol.fix(8203)
    m.fs.fwh_mixer[3].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[3].drain.unfix()

    _set_port(m.fs.fwh[3].inlet_1, m.fs.fwh_mixer[3].outlet)
    _set_port(m.fs.fwh[3].inlet_2, m.fs.fwh[2].outlet_2)
    m.fs.fwh[3].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh4
    _set_port(m.fs.fwh_mixer[4].steam,  m.fs.turbine_splitter[7].outlet_2)
    m.fs.fwh_mixer[4].drain.flow_mol.fix(1082)
    m.fs.fwh_mixer[4].drain.pressure.fix(351104)
    m.fs.fwh_mixer[4].drain.enth_mol.fix(10534)
    m.fs.fwh_mixer[4].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[4].drain.unfix()

    _set_port(m.fs.fwh[4].inlet_1, m.fs.fwh_mixer[4].outlet)
    _set_port(m.fs.fwh[4].inlet_2, m.fs.fwh[3].outlet_2)
    m.fs.fwh[4].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh5
    _set_port(m.fs.fwh[5].inlet_2, m.fs.fwh[4].outlet_2)
    _set_port(m.fs.fwh[5].inlet_1, m.fs.turbine_splitter[6].outlet_2)
    m.fs.fwh[5].initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  boiler feed pump and deaerator                                         #
    ###########################################################################
    # Deaerator
    _set_port(m.fs.deaerator.feedwater, m.fs.fwh[5].outlet_2)
    _set_port(m.fs.deaerator.steam, m.fs.turbine_splitter[5].outlet_2)
    m.fs.deaerator.drain.flow_mol[:].fix(4277)
    m.fs.deaerator.drain.pressure[:].fix(1379964)
    m.fs.deaerator.drain.enth_mol[:].fix(14898)
    m.fs.deaerator.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.deaerator.drain.unfix()

    # Booster pump
    _set_port(m.fs.booster.inlet, m.fs.deaerator.outlet)
    m.fs.booster.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  High-pressure feedwater heaters                                        #
    ###########################################################################
    # fwh6
    _set_port(m.fs.fwh_mixer[6].steam, m.fs.turbine_splitter[4].outlet_2)
    m.fs.fwh_mixer[6].drain.flow_mol.fix(4106)
    m.fs.fwh_mixer[6].drain.pressure.fix(2870602)
    m.fs.fwh_mixer[6].drain.enth_mol.fix(17959)
    m.fs.fwh_mixer[6].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[6].drain.unfix()

    _set_port(m.fs.fwh[6].inlet_1, m.fs.fwh_mixer[6].outlet)
    _set_port(m.fs.fwh[6].inlet_2, m.fs.booster.outlet)
    m.fs.fwh[6].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh7
    _set_port(m.fs.fwh_mixer[7].steam, m.fs.turbine_splitter[3].outlet_2)
    m.fs.fwh_mixer[7].drain.flow_mol.fix(3640)
    m.fs.fwh_mixer[7].drain.pressure.fix(4713633)
    m.fs.fwh_mixer[7].drain.enth_mol.fix(20472)
    m.fs.fwh_mixer[7].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[7].drain.unfix()

    _set_port(m.fs.fwh[7].inlet_1, m.fs.fwh_mixer[7].outlet)
    _set_port(m.fs.fwh[7].inlet_2, m.fs.fwh[6].outlet_2)
    m.fs.fwh[7].initialize(outlvl=outlvl, optarg=solver.options)

    # Boiler feed pump
    _set_port(m.fs.bfp.inlet, m.fs.fwh[7].outlet_2)
    m.fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh8
    _set_port(m.fs.fwh_mixer[8].steam, m.fs.turbine_splitter[2].outlet_2)
    m.fs.fwh_mixer[8].drain.flow_mol.fix(1311)
    m.fs.fwh_mixer[8].drain.pressure.fix(10282256)
    m.fs.fwh_mixer[8].drain.enth_mol.fix(25585)
    m.fs.fwh_mixer[8].initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh_mixer[8].drain.unfix()

    _set_port(m.fs.fwh[8].inlet_1, m.fs.fwh_mixer[8].outlet)
    _set_port(m.fs.fwh[8].inlet_2, m.fs.fwh[7].outlet_2)
    m.fs.fwh[8].initialize(outlvl=outlvl, optarg=solver.options)

    # fwh9
    _set_port(m.fs.fwh[9].inlet_2, m.fs.fwh[8].outlet_2)
    _set_port(m.fs.fwh[9].inlet_1, m.fs.turbine_splitter[1].outlet_2)
    m.fs.fwh[9].initialize(outlvl=outlvl, optarg=solver.options)

    #########################################################################
    #  Model Initialization with Square Problem Solve                       #
    #########################################################################
    #  Unfix split fractions and activate vapor fraction constraints
    #  Vaporfrac constraints set condensed steam enthalpy at the condensing
    #  side outlet to be that of a saturated liquid
    # Then solve the square problem again for an initilized model
    for i in pyo.RangeSet(10):
        m.fs.turbine_splitter[i].split_fraction[:, "outlet_2"].unfix()

    # keeping the extraction to deareator to be fixed
    # unfixing the extraction to bfpt
    m.fs.turbine_splitter[5].split_fraction[:, "outlet_2"].fix()
    m.fs.turbine_splitter[6].split_fraction[:, "outlet_3"].unfix()

    m.fs.constraint_bfp_power.activate()
    m.fs.constraint_out_pressure.activate()
    for j in pyo.RangeSet(9):
        m.fs.fwh[j].fwh_vfrac_constraint.activate()

    res = solver.solve(m)
    print("Model Initialization = ",
          res.solver.termination_condition)
    print("*********************Model Initialized**************************")
    
    return solver

def view_result(outfile, m):
    tags = {}

    ## Boiler
    tags['power_out'] = ("%4.2f" % pyo.value(m.fs.plant_power_out[0]))
    
    tags['boiler_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.boiler.inlet.flow_mol[0])*1e-3))
    tags['boiler_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.boiler.control_volume.properties_in[0].temperature)))
    tags['boiler_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.inlet.pressure[0])*1e-6))
    tags['boiler_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.inlet.enth_mol[0])*1e-3))
    tags['boiler_xin'] = ("%4.4f" % (pyo.value(
        m.fs.boiler.control_volume.properties_in[0].vapor_frac)))
    tags['boiler_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.boiler.outlet.flow_mol[0])*1e-3))
    tags['boiler_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.boiler.control_volume.properties_out[0].temperature)))
    tags['boiler_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.outlet.pressure[0])*1e-6))
    tags['boiler_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.boiler.outlet.enth_mol[0])*1e-3))
    tags['boiler_xout'] = ("%4.4f" % (pyo.value(
        m.fs.boiler.control_volume.properties_out[0].vapor_frac)))

    ## Reheater 1 & 2
    tags['turb3_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[3].inlet.flow_mol[0])*1e-3))
    tags['turb3_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[3].control_volume.properties_in[0].temperature)))
    tags['turb3_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[3].inlet.pressure[0])*1e-6))
    tags['turb3_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[3].inlet.enth_mol[0])*1e-3))
    tags['turb3_xin'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[3].control_volume.properties_in[0].vapor_frac)))

    tags['turb5_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[5].inlet.flow_mol[0])*1e-3))
    tags['turb5_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[5].control_volume.properties_in[0].temperature)))
    tags['turb5_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[5].inlet.pressure[0])*1e-6))
    tags['turb5_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[5].inlet.enth_mol[0])*1e-3))
    tags['turb5_xin'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[5].control_volume.properties_in[0].vapor_frac)))

    ## Turbine out
    tags['turb11_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.turbine[11].outlet.flow_mol[0])*1e-3))
    tags['turb11_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.turbine[11].control_volume.properties_out[0].temperature)))
    tags['turb11_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[11].outlet.pressure[0])*1e-6))
    tags['turb11_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.turbine[11].outlet.enth_mol[0])*1e-3))
    tags['turb11_xout'] = ("%4.4f" % (pyo.value(
        m.fs.turbine[11].control_volume.properties_out[0].vapor_frac)))

    ## Condenser
    tags['cond_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.condenser.outlet.flow_mol[0])*1e-3))
    tags['cond_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.condenser.control_volume.properties_out[0].temperature)))
    tags['cond_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.condenser.outlet.pressure[0])*1e-6))
    tags['cond_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.condenser.outlet.enth_mol[0])*1e-3))
    tags['cond_xout'] = ("%4.4f" % (pyo.value(
        m.fs.condenser.control_volume.properties_out[0].vapor_frac)))

    ## Feed water heaters
    tags['fwh9shell_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.flow_mol[0])*1e-3))
    tags['fwh9shell_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[9].shell.properties_in[0].temperature)))
    tags['fwh9shell_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.pressure[0])*1e-6))
    tags['fwh9shell_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[9].shell_inlet.enth_mol[0])*1e-3))
    tags['fwh9shell_xin'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[9].shell.properties_in[0].vapor_frac)))

    tags['fwh7tube_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.flow_mol[0])*1e-3))
    tags['fwh7tube_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[7].tube.properties_out[0].temperature)))
    tags['fwh7tube_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.pressure[0])*1e-6))
    tags['fwh7tube_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[7].tube_outlet.enth_mol[0])*1e-3))
    tags['fwh7tube_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[7].tube.properties_out[0].vapor_frac)))

    tags['fwh6shell_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.flow_mol[0])*1e-3))
    tags['fwh6shell_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[6].shell.properties_out[0].temperature)))
    tags['fwh6shell_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.pressure[0])*1e-6))
    tags['fwh6shell_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[6].shell_outlet.enth_mol[0])*1e-3))
    tags['fwh6shell_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[6].shell.properties_out[0].vapor_frac)))

    tags['fwh5tube_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.flow_mol[0])*1e-3))
    tags['fwh5tube_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[5].tube.properties_out[0].temperature)))
    tags['fwh5tube_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.pressure[0])*1e-6))
    tags['fwh5tube_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].tube_outlet.enth_mol[0])*1e-3))
    tags['fwh5tube_xout'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[5].tube.properties_out[0].vapor_frac)))

    tags['fwh5shell_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.flow_mol[0])*1e-3))
    tags['fwh5shell_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.fwh[5].shell.properties_in[0].temperature)))
    tags['fwh5shell_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.pressure[0])*1e-6))
    tags['fwh5shell_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.fwh[5].shell_inlet.enth_mol[0])*1e-3))
    tags['fwh5shell_xin'] = ("%4.4f" % (pyo.value(
        m.fs.fwh[5].shell.properties_in[0].vapor_frac)))

    ## Deareator
    tags['da_steam_Fin'] = ("%4.3f" % (pyo.value(
        m.fs.deaerator.steam.flow_mol[0])*1e-3))
    tags['da_steam_Tin'] = ("%4.2f" % (pyo.value(
        m.fs.deaerator.steam_state[0].temperature)))
    tags['da_steam_Pin'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.steam.pressure[0])*1e-6))
    tags['da_steam_Hin'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.steam.enth_mol[0])*1e-3))
    tags['da_steam_xin'] = ("%4.4f" % (pyo.value(
        m.fs.deaerator.steam_state[0].vapor_frac)))
    tags['da_Fout'] = ("%4.3f" % (pyo.value(
        m.fs.deaerator.outlet.flow_mol[0])*1e-3))
    tags['da_Tout'] = ("%4.2f" % (pyo.value(
        m.fs.deaerator.mixed_state[0].temperature)))
    tags['da_Pout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.outlet.pressure[0])*1e-6))
    tags['da_Hout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.outlet.enth_mol[0])*1e-3))
    tags['da_xout'] = ("%4.1f" % (pyo.value(
        m.fs.deaerator.mixed_state[0].vapor_frac)))

    ## Feed water heaters mixers

    for i in m.mixer_list:
        tags['fwh'+str(i)+'mix_steam_Fin'] = ("%4.3f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.flow_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_steam_Tin'] = ("%4.2f" % (pyo.value(
            m.fs.fwh_mixer[i].steam_state[0].temperature)))
        tags['fwh'+str(i)+'mix_steam_Pin'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.pressure[0])*1e-6))
        tags['fwh'+str(i)+'mix_steam_Hin'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].steam.enth_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_steam_xin'] = ("%4.4f" % (pyo.value(
            m.fs.fwh_mixer[i].steam_state[0].vapor_frac)))
        tags['fwh'+str(i)+'mix_Fout'] = ("%4.3f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.flow_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_Tout'] = ("%4.2f" % (pyo.value(
            m.fs.fwh_mixer[i].mixed_state[0].temperature)))
        tags['fwh'+str(i)+'mix_Pout'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.pressure[0])*1e-6))
        tags['fwh'+str(i)+'mix_Hout'] = ("%4.1f" % (pyo.value(
            m.fs.fwh_mixer[i].outlet.enth_mol[0])*1e-3))
        tags['fwh'+str(i)+'mix_xout'] = ("%4.4f" % (pyo.value(
            m.fs.fwh_mixer[i].mixed_state[0].vapor_frac)))

    # BFP
    tags['bfp_power'] = ("%4.2f" % (pyo.value(
        m.fs.bfp.control_volume.work[0])*1e-6))
    tags['booster_power'] = ("%4.2f" % (pyo.value(
        m.fs.booster.control_volume.work[0])*1e-6))
    tags['bfpt_power'] = ("%4.2f" % (pyo.value(
        m.fs.bfpt.control_volume.work[0])*-1e-6))
    tags['cond_power'] = ("%4.2f" % (pyo.value(
        m.fs.cond_pump.control_volume.work[0])*1e-6))

    original_svg_file = os.path.join(
        this_file_dir(), "pfd_ultra_supercritical_pc.svg")
    with open(original_svg_file, "r") as f:
        svg_tag(tags, f, outfile=outfile)


def build_plant_model(initialize_from_file=None, store_initialization=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = declare_unit_model()

    # Give all the required inputs to the model
    # Ensure that the degrees of freedom = 0 (model is complete)
    set_model_input(m)
    # Assert that the model has no degree of freedom at this point
    assert degrees_of_freedom(m) == 0

    set_scaling_factors(m)
    # Initialize the model (sequencial initialization and custom routines)
    # Ensure after the model is initialized, the degrees of freedom = 0
    solver = initialize(m)
    assert degrees_of_freedom(m) == 0

    # The power plant with storage for a charge scenario is now ready
    #  Declaraing a plant power out variable for easy analysis of various
    #  design and operating scenarios
    m.fs.plant_power_out = pyo.Var(
        m.fs.time,
        domain=pyo.Reals,
        initialize=400,
        doc="Net Power MWe out from the power plant",
        units=pyunits.MW
    )

    #   Constraint on Plant Power Output
    #   Plant Power Out = Total Turbine Power
    @m.fs.Constraint(m.fs.time)
    def production_cons(b, t):
        return (
            (-1*sum(m.fs.turbine[p].work_mechanical[t]
                 for p in pyo.RangeSet(11))
             )
            == m.fs.plant_power_out[t]*1e6*(pyunits.W/pyunits.MW)
        )

    return m, solver


def model_analysis(m, solver):

#   Solving the flowsheet and check result
#   At this time one can make chnages to the model for further analysis
    flow_frac_list = [1.0]
    pres_frac_list = [1.0]
    for i in flow_frac_list:
        for j in pres_frac_list:
            m.fs.boiler.inlet.flow_mol.fix(i*17854)  # mol/s
            m.fs.boiler.outlet.pressure.fix(j*31125980)
            solver.solve(m, tee=True, symbolic_solver_labels=True)
            print('Plant Power (MW) =', pyo.value(m.fs.plant_power_out[0]))
    return m

if __name__ == "__main__":
    m, solver = build_plant_model(initialize_from_file=None,
                                  store_initialization=None)
    # User can import the model from build_plant_model for analysis
    # A sample analysis function is called below
    m_result = model_analysis(m, solver)
    # View results in a process flow diagram
    view_result("pfd_usc_powerplant_result.svg", m_result)

