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

__author__ = "Andres J Calderon, Naresh Susarla, Miguel Zamarripa, " \
             "Jaffer Ghouse, Radhakrishna Tumbalam Gooty"

import pandas as pd

# Import Pyomo libraries
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    RangeSet,
    ConstraintList,
    TransformationFactory,
    units as pyunits,
)
from pyomo.network import Arc

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import (
    HeatExchanger,
    PressureChanger,
    MomentumMixingType,
    Heater,
    Separator,
)
from idaes.models.unit_models.heat_exchanger import delta_temperature_underwood_callback

from idaes.models_extra.power_generation.unit_models.helm import (
    HelmSplitter,
    HelmTurbineStage,
    HelmMixer as Mixer,
    HelmIsentropicCompressor as WaterPump,
    HelmNtuCondenser as CondenserHelm,
)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.models.properties import iapws95 

from dispatches.unit_models import ConcreteTES


"""
This is a simple power plant model for supercritical coal-fired power plant
This model uses some of the generic unit models in place of complex
power generation unit models such as multi-stage turbine and 0D FWH.
For example, this model uses multiple pressure changers to
model multi-stage turbine and 0D heat exchangers to model feed water heaters.
Some of the parameters in the model such as feed water heater areas,
overall heat transfer coefficient, turbine efficiencies at multiple stages
have all been estimated in order to closely match with the data in
NETL baseline report.
Additional assumptions include using Heater blocks to model main steam boiler,
reheater, and condenser. Also, the throttle valves at turbine inlet are
not included in this model.
The model has 3 degrees of freedom to be specified by the user. These are the
conditions for main steam: Pressure (Pa), Temperature (K), and Flow (mol/s).
The default values fixed in the model are from the NETL Baseline report rev4.

"""

CONC_TES_DATA = {
    "num_tubes": 10000,
    "num_segments": 20,
    "num_time_periods": 2,
    "tube_length": 64.9,
    "tube_diameter": 0.0105664,
    "face_area": 0.00847,
    "therm_cond_concrete": 1,
    "dens_mass_concrete": 2240,
    "cp_mass_concrete": 900,
    "init_temperature_concrete": [
        750, 732.631579, 715.2631579, 697.8947368, 680.5263158, 663.1578947,
        645.7894737, 628.4210526, 611.0526316, 593.6842105, 576.3157895, 558.9473684,
        541.5789474, 524.2105263, 506.8421053, 489.4736842, 472.1052632, 454.7368421,
        437.3684211, 420
    ],
    "flow_mol_charge": 0.00958 * 1000 / 18.01528,
    "inlet_pressure_charge": 19600000,
    "inlet_temperature_charge": 865,
    "flow_mol_discharge": 3 / 18.01528,
    "inlet_pressure_discharge": 8.5e5, #15 MPa
    "inlet_temperature_discharge": 355
}
DIS_IN_PRES = 8.5e5
DIS_IN_TEMP = 355


def build_scpc_flowsheet(m=None, include_concrete_tes=True, conc_tes_data=CONC_TES_DATA):
    """
    Build model for a supercritical pulverized coal-fired power plant

    Args:
        m: Pyomo `ConcreteModel` or `Block`
        include_concrete_tes: If True, includes concrete thermal energy storage

    Returns:
        m
    """

    if m is None:
        m = ConcreteModel()
    
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.include_concrete_tes = include_concrete_tes

    # Add thermodynamic package.
    m.fs.prop_water_mix = iapws95.Iapws95ParameterBlock(
        phase_presentation=iapws95.PhaseType.MIX,
    )
    m.fs.prop_water_mix.default_scaling_factor["pressure", None] = 1e-5
    m.fs.prop_water_mix.default_scaling_factor["flow_mol", None] = 1e-2

    # Add sets to the model
    m.fs.set_turbines = RangeSet(9, doc="Set of turbines")
    m.fs.set_t_splitters = RangeSet(8, doc="Set of splitters")
    m.fs.set_fwh = Set(initialize=[1, 2, 3, 4, 6, 7, 8], doc="Set of feed water heaters")
    m.fs.set_fwh_mix = Set(initialize=[1, 2, 3, 5, 6, 7], doc="Set of mixers associated with fwh")

    """
    TURBINE TRAIN: This part builds, turbine models, splitters models
    after each turbine, and hp_splitter models
    """
    # Add HP steam splitter
    m.fs.hp_splitter = HelmSplitter(property_package=m.fs.prop_water_mix)

    # Add turbine stages
    m.fs.turbine = HelmTurbineStage(m.fs.set_turbines, property_package=m.fs.prop_water_mix)

    # Add boiler feed water turbine
    m.fs.bfpt = HelmTurbineStage(property_package=m.fs.prop_water_mix)

    # Add splitters associated with turbines. Except splitter 4,
    # all splitters have 2 outlets. Splitter 4 has three outlets
    m.fs.t_splitter = HelmSplitter(
        m.fs.set_t_splitters,
        property_package=m.fs.prop_water_mix,
        num_outlets=2,
        # Set the number of outlets for splitter 4
        initialize={4: {
            "property_package": m.fs.prop_water_mix,
            "num_outlets": 3,}
        },
    )

    """
    BOILER AND FEED WATER HEATERS: This part builds models for boiler, reheater,
    and feed water heaters.

    Feed water heaters (FWHs) are declared as 0D heat exchangers
    Tube side is for feed water & Shell side is for steam condensing
    Pressure drop on both sides are accounted for by setting the respective
    outlet pressure based on the following assumptions:
        (1) Feed water side: A constant 4% pressure drop is assumed
              on the feedwater side for all FWHs. For this,
              the outlet pressure is set to 0.96 times the inlet pressure,
              on the feed water side for all FWHs
        (2) Steam condensing side: Going from high pressure to
              low pressure FWHs, the outlet pressure of
              the condensed steam in assumed to be 10% more than that
              of the pressure of steam extracted for the immediately
              next lower pressure feedwater heater.
              e.g. the outlet condensate pressure of FWH 'n'
              = 1.1 * pressure of steam extracted for FWH 'n-1'
              In case of FWH1 the FWH 'n-1' is used for Condenser,
              and in case of FWH6, FWH 'n-1' is for Deaerator. Here,
              the steam pressure for FWH 'n-1' is known because the
              pressure ratios for turbines are fixed.
    The condensing steam is assumed to leave the FWH as saturated liquid
    Thus, each FWH is accompanied by 3 constraints, 2 for pressure drop
    and 1 for the enthalpy.

    Boiler section is set up using two heater blocks, as following:
    1) For the main steam the heater block is named 'boiler'
    2) For the reheated steam the heater block is named 'reheater'
    """
    # Add boiler
    m.fs.boiler = Heater(
        property_package=m.fs.prop_water_mix, 
        has_pressure_change=True,
    )

    # Add reheater
    m.fs.reheater = Heater(
        property_package=m.fs.prop_water_mix,
        has_pressure_change=True,
    )

    # Outlet temperature of boiler is set to 866.15 K
    @m.fs.boiler.Constraint(m.fs.time)
    def boiler_temperature_constraint(blk, t):
        return blk.control_volume.properties_out[t].temperature == 866.15  # K

    # Outlet temperature of reheater is set to 866.15 K
    @m.fs.reheater.Constraint(m.fs.time)
    def reheater_temperature_constraint(blk, t):
        return blk.control_volume.properties_out[t].temperature == 866.15  # K

    # Add feed water heaters
    m.fs.fwh = HeatExchanger(
        m.fs.set_fwh,
        delta_temperature_callback=delta_temperature_underwood_callback,
        hot_side_name="shell",
        cold_side_name="tube",
        shell={
            "property_package": m.fs.prop_water_mix,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "has_pressure_change": True,
        },
        tube={
            "property_package": m.fs.prop_water_mix,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "has_pressure_change": True,
        },
    )

    # Data needed to set the pressure change on side 1 (shell) of fwh
    # FWH1: 0.5 is the pressure ratio for turbine #9 (see set_inputs)
    # FWH2: 0.64^2 is the pressure ratio for turbine #8 (see set_inputs)
    # FWH3: 0.64^2 is the pressure ratio for turbine #7 (see set_inputs)
    # FWH4: 0.64^2 is the pressure ratio for turbine #6 (see set_inputs)
    # FWH6: 0.79^6 is the pressure ratio for turbine #4 (see set_inputs)
    # FWH7: 0.79^4 is the pressure ratio for turbine #3 (see set_inputs)
    # FWH8: 0.8^2 is the pressure ratio for turbine #2 (see set_inputs)

    pressure_ratio_list = {1: 0.5,
                           2: 0.64 ** 2,
                           3: 0.64 ** 2,
                           4: 0.64 ** 2,
                           6: 0.79 ** 6,
                           7: 0.79 ** 4,
                           8: 0.8 ** 2}
    for i in m.fs.set_fwh:
        obj = m.fs.fwh[i]

        # Outlet enthalpy of condensate in an FWH must be same as that of saturated liquid
        @obj.Constraint(m.fs.time)
        def vapor_frac_constraint(blk, t):
            return (blk.hot_side.properties_out[t].enth_mol ==
                    blk.hot_side.properties_out[t].enth_mol_sat_phase['Liq'])

        # Setting a 4% pressure drop on the feedwater side (P_out = 0.96 * P_in)
        @obj.Constraint(m.fs.time)
        def s2_pdrop_constraint(blk, t):
            return (blk.cold_side.properties_out[t].pressure ==
                    0.96 * blk.cold_side.properties_in[t].pressure)

        # Setting the outlet pressure of condensate to be 10% more than that of
        # steam routed to condenser, as described in FWH description
        @obj.Constraint(m.fs.time)
        def s1_pdrop_constraint(blk, t):
            return (blk.hot_side.properties_out[t].pressure ==
                    1.1 * pressure_ratio_list[i] * blk.hot_side.properties_in[t].pressure)

    """
    FEED WATER MIXERS: This part adds mixers associated with feed water heaters
    """
    # Mixer 5 has an additional inlet from feedwater heater, and Mixer 7 has an
    # additional inlet from hx_pump
    m.fs.fwh_mix = Mixer(
        m.fs.set_fwh_mix,
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["steam", "drain"],
        property_package=m.fs.prop_water_mix,
        initialize={
            # Add feedwater inlet to the deaerator
            5: {"momentum_mixing_type": MomentumMixingType.none,
                "inlet_list": ["steam", "drain", "feedwater"],
                "property_package": m.fs.prop_water_mix},
            # Add an inlet from hx_pump to mixer 7
            7: {"momentum_mixing_type": MomentumMixingType.none,
                "inlet_list": ["steam", "drain", "from_storage"],
                "property_package": m.fs.prop_water_mix},
            },
        )

    # Pressure constraint for all mixers
    for i in m.fs.set_fwh_mix:
        obj = m.fs.fwh_mix[i]

        if i == 5:  # Deaerator
            # Since the deaerator's (fwh_mix[5]) 'feedwater' inlet has the
            # minimum pressure, the outlet pressure must be the same as that
            # of the 'feedwater' inlet

            @obj.Constraint(m.fs.time)
            def mixer_pressure_constraint(blk, t):
                return blk.mixed_state[t].pressure == blk.feedwater_state[t].pressure

        else:
            # For the remaining mixers, the steam from the splitters has the minimum
            # pressure, so the outlet pressure must be the same as that of inlet steam

            @obj.Constraint(m.fs.time)
            def mixer_pressure_constraint(blk, t):
                return blk.mixed_state[t].pressure == blk.steam_state[t].pressure

    """
    CONDENSATION STAGE: This part adds models for all the unit operations 
    related to the heat removal process.
    """
    # Add condenser mixer
    # The inlet 'main' refers to the main steam coming from the turbine train
    # Inlet 'bfpt' refers to the steam coming from the bolier feed pump turbine
    # Inlet 'drain' refers to the condensed steam from the feed water heater 1
    # Inlet 'makeup' refers to the make up water
    m.fs.condenser_mix = Mixer(
        momentum_mixing_type=MomentumMixingType.none,
        inlet_list=["main", "bfpt", "drain", "makeup"],
        property_package=m.fs.prop_water_mix,
    )

    # The outlet pressure of condenser mixer is equal to the minimum pressure
    # Since the turbine (#9) outlet (or, mixer inlet 'main') pressure
    # has the minimum pressure, the following constraint sets the outlet
    # pressure of the condenser mixer to the pressure of the inlet 'main'
    @m.fs.condenser_mix.Constraint(m.fs.time)
    def mixer_pressure_constraint(blk, t):
        return blk.mixed_state[t].pressure == blk.main_state[t].pressure

    # Add condenser
    m.fs.condenser = CondenserHelm(
        shell={"has_pressure_change": False, "property_package": m.fs.prop_water_mix},
        tube={"has_pressure_change": False, "property_package": m.fs.prop_water_mix},
    )

    """
    PUMPS: This part adds models for all the pumps used in the flowsheet
    """
    # Add condensate pump (Pumps water from the condenser outlet)
    m.fs.cond_pump = WaterPump(property_package=m.fs.prop_water_mix)

    # Add boiler feed water pump (Pumps water between feed water heaters)
    m.fs.bfp = WaterPump(property_package=m.fs.prop_water_mix)

    # Add a splitter before bfp to extract water for the discharge
    m.fs.bfp_splitter = HelmSplitter(property_package=m.fs.prop_water_mix)

    if include_concrete_tes:
        append_tes_unit_models(m, conc_tes_data)

    else:
        # Concrete TES is not included, so close the ports
        # in hp_splitter, bfp_splitter, fwh_mix[7]
        m.fs.hp_splitter.split_fraction[:, "outlet_2"].fix(0)
        m.fs.bfp_splitter.split_fraction[:, "outlet_2"].fix(0)
        m.fs.fwh_mix[7].from_storage.flow_mol.fix(0)
        m.fs.fwh_mix[7].from_storage.pressure.fix(24235081.4)
        m.fs.fwh_mix[7].from_storage.enth_mol.fix(40000)

    """
    This part adds flowsheet-level constraints
    """
    # The following constraint sets the outlet pressure of steam extracted
    # for boiler feed water turbine to be same as that of condenser
    @m.fs.Constraint(m.fs.time)
    def constraint_bfpt_out_pressure(blk, t):
        return (blk.bfpt.control_volume.properties_out[t].pressure ==
                blk.condenser_mix.mixed_state[t].pressure)

    # The following constraint demands that the work done by the
    # boiler feed water pump is same as that of boiler feed water turbine
    # Essentially, this says that boiler feed water turbine produces just
    # enough power to meet the demand of boiler feed water pump
    @m.fs.Constraint(m.fs.time)
    def constraint_bfp_power(blk, t):
        return blk.bfp.control_volume.work[t] + blk.bfpt.control_volume.work[t] == 0

    # Declare net power output variable
    m.fs.net_power_output = Var(
        m.fs.time,
        initialize=620 * 1e6,
        doc="Net Power We out from the power plant",
        units=pyunits.W,
    )

    #   Constraint on Plant Power Output
    #   Plant Power Out = Turbine Power - Power required for HX Pump
    @m.fs.Constraint(m.fs.time)
    def production_cons(blk, t):
        return (sum(blk.turbine[j].work_mechanical[t] for j in m.fs.set_turbines)
                + m.fs.cond_pump.control_volume.work[t] == -m.fs.net_power_output[t])

    create_arcs(m)

    return m


def append_tes_unit_models(m, tes_data):
    """
    STORAGE: This function adds TES model and the models for auxiliary
    equipment required for the TES unit
    """

    m.fs.tes = ConcreteTES(
        model_data=tes_data,
        property_package=m.fs.prop_water_mix,
        operating_mode="combined",
    )

    m.fs.hp_splitter_to_tes = Arc(source=m.fs.hp_splitter.outlet_2,
                                  destination=m.fs.tes.inlet_charge)
    m.fs.tes_to_fwh_mix_7 = Arc(source=m.fs.tes.outlet_charge,
                                destination=m.fs.fwh_mix[7].from_storage)

    # Connect the outlet_2 of bfp_splitter to the cold side of tes
    inlet_enthalpy_discharge = iapws95.htpx(T=DIS_IN_TEMP * pyunits.K, P=DIS_IN_PRES * pyunits.Pa)
    m.fs.tes.inlet_discharge.pressure.fix(DIS_IN_PRES)
    m.fs.tes.inlet_discharge.enth_mol.fix(inlet_enthalpy_discharge)

    # @m.fs.Constraint(m.fs.time)
    # def discharge_flow_constraint(blk, t):
    #     return blk.tes.inlet_discharge.flow_mol[t] <= 0.2 * blk.bfp_splitter.inlet.flow_mol[t]

    m.fs.bfp_splitter.split_fraction[:, "outlet_2"].fix(0)

    """
    DISCHARGE TRAIN: This part adds the turbine for the discharge 
    process, and other associated equipment.
    """
    m.fs.discharge_turbine = HelmTurbineStage(property_package=m.fs.prop_water_mix)

    # To avoid condensation inside the turbine, we impose the following
    # constraint on the outlet
    @m.fs.discharge_turbine.Constraint(m.fs.time)
    def turbine_enthalpy_constraint(blk, t):
        # return (blk.control_volume.properties_out[t].enth_mol ==
        #         blk.control_volume.properties_out[t].enth_mol_sat_phase['Vap'])
        # return blk.control_volume.properties_out[t].temperature == 310
        return blk.control_volume.properties_out[t].pressure == 6644

    m.fs.tes_to_dis_turbine = Arc(source=m.fs.tes.outlet_discharge,
                                  destination=m.fs.discharge_turbine.inlet)

    return


def create_arcs(m):
    """
    TURBINE TRAIN: This part connects models in turbine train
    """
    # The following dictionary contains information on the destination
    # of outlet_2 stream from t_splitter.
    split_fwh_map = {1: ["fwh", 8],
                     2: ["fwh_mix", 7],
                     3: ["fwh_mix", 6],
                     4: ["fwh_mix", 5],  # Deaerator
                     5: ["fwh", 4],
                     6: ["fwh_mix", 3],
                     7: ["fwh_mix", 2],
                     8: ["fwh_mix", 1]}

    for i in m.fs.set_t_splitters:
        # Add arc connecting turbine and splitter
        setattr(m.fs, "turbine_" + str(i) + "_to_splitter_" + str(i),
                Arc(source=m.fs.turbine[i].outlet,
                    destination=m.fs.t_splitter[i].inlet))

        # Add arc connecting the splitter's outlet_1 to turbine/reheater
        if i == 2:
            # First outlet from splitter 2 goes to the reheater
            setattr(m.fs, "splitter_" + str(i) + "_to_reheater",
                    Arc(source=m.fs.t_splitter[i].outlet_1,
                        destination=m.fs.reheater.inlet))
        else:
            setattr(m.fs, "splitter_" + str(i) + "_to_turbine_" + str(i + 1),
                    Arc(source=m.fs.t_splitter[i].outlet_1,
                        destination=m.fs.turbine[i + 1].inlet))

        # Add arc connecting the splitter's outlet_2 to fwh/fwh_mix
        if split_fwh_map[i][0] == "fwh":
            setattr(m.fs, "splitter_" + str(i) + "_to_fwh_" + str(split_fwh_map[i][1]),
                    Arc(source=m.fs.t_splitter[i].outlet_2,
                        destination=m.fs.fwh[split_fwh_map[i][1]].shell_inlet))
        else:
            setattr(m.fs, "splitter_" + str(i) + "_to_fwh_mix_" + str(split_fwh_map[i][1]),
                    Arc(source=m.fs.t_splitter[i].outlet_2,
                        destination=m.fs.fwh_mix[split_fwh_map[i][1]].steam))

    # Connect the third outlet from splitter 4 to bfpt
    m.fs.splitter_4_to_bfpt = Arc(source=m.fs.t_splitter[4].outlet_3,
                                  destination=m.fs.bfpt.inlet)

    """
    BOILER: This part connects models related to boilers/feed water heaters
    """
    m.fs.boiler_to_hp_splitter = Arc(source=m.fs.boiler.outlet,
                                     destination=m.fs.hp_splitter.inlet)

    m.fs.hp_splitter_to_turbine_1 = Arc(source=m.fs.hp_splitter.outlet_1,
                                        destination=m.fs.turbine[1].inlet)

    m.fs.reheater_to_turbine_3 = Arc(source=m.fs.reheater.outlet,
                                     destination=m.fs.turbine[3].inlet)

    for i in m.fs.set_fwh_mix:  # List of fwh mixers
        # Add arcs to connect the outlet of fwh_mix to fwh/bfp
        if i == 5:
            # Outlet of Deaerator goes to bfp_splitter
            m.fs.fwh_mix_5_to_bfp_splitter = Arc(source=m.fs.fwh_mix[5].outlet,
                                                 destination=m.fs.bfp_splitter.inlet)

            # Outlet_1 of bfp_splitter goes to bfp
            m.fs.bfp_splitter_to_bfp = Arc(source=m.fs.bfp_splitter.outlet_1,
                                           destination=m.fs.bfp.inlet)

            # Boiler feed water pump to feed water heater 6
            m.fs.bfp_to_fwh_6 = Arc(source=m.fs.bfp.outlet,
                                    destination=m.fs.fwh[6].tube_inlet)

        else:
            # Outlet of each fwh_mix goes to the shell_inlet of fwh
            setattr(m.fs, "fwh_mix_" + str(i) + "_to_fwh_" + str(i),
                    Arc(source=m.fs.fwh_mix[i].outlet,
                        destination=m.fs.fwh[i].shell_inlet))

        # Add arcs connecting the drains and the mixers
        setattr(m.fs, "fwh_" + str(i + 1) + "_to_fwh_mix_" + str(i),
                Arc(source=m.fs.fwh[i + 1].shell_outlet,
                    destination=m.fs.fwh_mix[i].drain))

        # Add arcs connecting feed water outlet to feed water inlet
        if i != 5:  # Skip Deaerator
            setattr(m.fs, "fwh_" + str(i) + "_to_fwh_" + str(i + 1),
                    Arc(source=m.fs.fwh[i].tube_outlet,
                        destination=m.fs.fwh[i + 1].tube_inlet))

    m.fs.fwh_4_to_fwh_mix_5 = Arc(source=m.fs.fwh[4].tube_outlet,
                                  destination=m.fs.fwh_mix[5].feedwater)

    m.fs.fwh_8_to_boiler = Arc(source=m.fs.fwh[8].tube_outlet,
                               destination=m.fs.boiler.inlet)

    """
    CONDENSER: This part connects models related to the heat removal process
    """
    # Last turbine outlet to the mixer associated with the condenser
    m.fs.turbine_9_to_condenser_mix = Arc(source=m.fs.turbine[9].outlet,
                                          destination=m.fs.condenser_mix.main)

    # FWH to the mixer associated with condenser
    m.fs.fwh_1_to_condenser_mix = Arc(source=m.fs.fwh[1].shell_outlet,
                                      destination=m.fs.condenser_mix.drain)

    # Boiler feed water turbine to the mixer associated with condenser
    m.fs.bfpt_to_condenser_mix = Arc(source=m.fs.bfpt.outlet,
                                     destination=m.fs.condenser_mix.bfpt)

    # Mixer associated with condenser to condenser
    m.fs.condenser_mix_to_condenser = Arc(source=m.fs.condenser_mix.outlet,
                                          destination=m.fs.condenser.shell_inlet)

    # Condenser outlet to condenser pump
    m.fs.condenser_to_pump = Arc(source=m.fs.condenser.shell_outlet,
                                 destination=m.fs.cond_pump.inlet)

    # Condenser pump outlet to feed water heater
    m.fs.cond_pump_to_fwh_1 = Arc(source=m.fs.cond_pump.outlet,
                                  destination=m.fs.fwh[1].tube_inlet)

    TransformationFactory("network.expand_arcs").apply_to(m.fs)


def fix_dof_and_initialize(
    m, 
    outlvl=idaeslog.INFO, 
    hp_split_fraction=0.1,
    discharge_flow = 1,
):
    """
    Model inputs / fixed variable or parameter values
    assumed in this block, unless otherwise stated explicitly,
    are either assumed or estimated in order to match the results with
    known baseline scenario for supercritical steam cycle

    1. Each turbine has two dof: pressure ratio and isentropic efficiency
    2. Boiler feed water turbine has one dof: isentropic efficiency. The other
       dof is imposed as a constraint that requires the outlet pressure to be
       the same as the pressure of the condenser mixer outlet.
    3. t_splitters DO NOT have any dof. The split fraction is governed by the
       constraint that requires the outlet_1 of each fwh to be at saturated
       liquid state. However, since splitter 4 outlet does not go to a fwh, the
       above constraint is not applicable for it. So, we set the split fraction
       of the stream going to fwh_mix[5] i.e., deaerator. The other split fraction
       is governed by the constraint that -bfpt_power = bfp_power. In short, of the
       8 t_splitters, t_splitter[4] has one dof and the rest do not have any dof.
    4. hp_splitter and ip splitter have one dof: split fraction
    5. Boiler has two dof: outlet temperature and pressure. In addition to
       these two, we specify the inlet molar flowrate and remove one dof
       from condenser_mix
    6. Reheater has two dof: outlet temperature and pressure drop
    7. feed water heaters have two dof: area and heat transfer coefficient
    8. mixers associated with feed water heaters DO NOT have any dof
    9. Condenser mixer has three dof: state of the makeup. However, we will not
       fix the molar flowrate since we fixed an extra variable in boiler model
    10. Condenser has four dof: pressure and molar enthalpy of tube_inlet, area and
        heat transfer coefficient. The condenser model has a constraint that
        governs the molar flowrate of tube_inlet.
    11. Condenser pump has two dof: deltaP and isentropic efficiency
    12. Boiler feed water pump has two dof: outlet pressure and isentropic efficiency
    13. hx_pump has two dof: outlet pressure and isentropic efficiency
    14. Storage cooler has one dof. The other dof is imposed as a constraint
        on the outlet state (Outlet is at saturated liquid state).
    """

    main_steam_pressure = 24235081.4  # Pa
    boiler_inlet_flow_mol = 29111  # mol/s
    reheater_pressure_drop = -96526.64  # Pa, based on NETL Baseline report

    turbine_dof = {1: {"pr": 0.80 ** 5, "eta": 0.94},
                   2: {"pr": 0.80 ** 2, "eta": 0.94},
                   3: {"pr": 0.79 ** 4, "eta": 0.88},
                   4: {"pr": 0.79 ** 6, "eta": 0.88},
                   5: {"pr": 0.64 ** 2, "eta": 0.78},
                   6: {"pr": 0.64 ** 2, "eta": 0.78},
                   7: {"pr": 0.64 ** 2, "eta": 0.78},
                   8: {"pr": 0.64 ** 2, "eta": 0.78},
                   9: {"pr": 0.50, "eta": 0.78}}

    fwh_dof = {1: {"area": 400, "htc": 2000},
               2: {"area": 300, "htc": 2900},
               3: {"area": 200, "htc": 2900},
               4: {"area": 200, "htc": 2900},
               6: {"area": 600, "htc": 2900},
               7: {"area": 400, "htc": 2900},
               8: {"area": 400, "htc": 2900}}

    # Fix the degrees of freedom of turbines
    for i in turbine_dof:
        m.fs.turbine[i].ratioP.fix(turbine_dof[i]["pr"])
        m.fs.turbine[i].efficiency_isentropic.fix(turbine_dof[i]["eta"])

    # Boiler feed water turbine. The outlet pressure is set as a constraint
    m.fs.bfpt.efficiency_isentropic.fix(0.80)

    # Splitters
    m.fs.t_splitter[4].split_fraction[:, "outlet_2"].fix(0.050331)

    if m.fs.include_concrete_tes:
        m.fs.hp_splitter.split_fraction[:, "outlet_2"].fix(hp_split_fraction)
        m.fs.bfp_splitter.split_fraction[:, "outlet_2"].fix(0)

        # Fix degrees of freedom of the discharge turbine. we need to fix
        # only one dof for this turbine. The other dof is specified as a
        # constraint on the state of the outlet.
        m.fs.discharge_turbine.efficiency_isentropic.fix(0.75)

    # Fix the degrees of freedom of boiler
    m.fs.boiler.inlet.flow_mol.fix(boiler_inlet_flow_mol)
    m.fs.boiler.outlet.pressure.fix(main_steam_pressure)

    # Fix the degrees of freedom of reheater
    m.fs.reheater.deltaP.fix(reheater_pressure_drop)

    # Fix the degrees of freedom of feed water heaters
    for i in fwh_dof:
        m.fs.fwh[i].area.fix(fwh_dof[i]["area"])
        m.fs.fwh[i].overall_heat_transfer_coefficient.fix(fwh_dof[i]["htc"])

    # Make up stream to condenser
    m.fs.condenser_mix.makeup.flow_mol.value = 1.08002495835536E-12  # mol/s
    m.fs.condenser_mix.makeup.pressure.fix(103421.4)  # Pa
    m.fs.condenser_mix.makeup.enth_mol.fix(1131.69204)  # J/mol

    # Condenser
    m.fs.condenser.tube_inlet.pressure.fix(500000)
    m.fs.condenser.tube_inlet.enth_mol.fix(1800)
    m.fs.condenser.area.fix(34000)
    m.fs.condenser.overall_heat_transfer_coefficient.fix(3100)

    # Condenser pump
    m.fs.cond_pump.efficiency_isentropic.fix(0.80)
    m.fs.cond_pump.deltaP.fix(1e6)

    # Boiler feed water pump. Outlet pressure is assumed to be 15% more than the
    # desired main steam (Turbine Inlet) pressure to account for the pressure drop
    # across Feed water heaters and Boiler
    m.fs.bfp.efficiency_isentropic.fix(0.80)
    m.fs.bfp.outlet.pressure.fix(main_steam_pressure * 1.15)  # Pa

    """
    Initialization Begins
    """
    # Some numbers useful for initialization, but THESE ARE NOT DOF. We fix these
    # quantities at the beginning of the initialization process and unfix them
    # once the flowsheet has been initialized
    # FIXME: Had to change the value of fwh_mix_drain_state[1]["flow"]
    boiler_inlet_state = {"pressure": 24657896, "enthalpy": 20004}
    condenser_drain_state = {"flow": 1460, "pressure": 7308, "enthalpy": 2973}
    fwh_mix_drain_state = {
        1: {"flow": 1434 * 1.1, "pressure": 14617, "enthalpy": 3990},
        2: {"flow": 1136, "pressure": 35685, "enthalpy": 5462},
        3: {"flow": 788, "pressure": 87123, "enthalpy": 7160},
        5: {"flow": 6207, "pressure": 519291, "enthalpy": 11526},
        6: {"flow": 5299, "pressure": 2177587, "enthalpy": 16559},
        7: {"flow": 3730, "pressure": 5590711, "enthalpy": 21232},
    }

    # Fixing the split fractions of the splitters. Will be unfixed later
    m.fs.t_splitter[1].split_fraction[:, "outlet_2"].fix(0.12812)
    m.fs.t_splitter[2].split_fraction[:, "outlet_2"].fix(0.061824)
    m.fs.t_splitter[3].split_fraction[:, "outlet_2"].fix(0.03815)
    m.fs.t_splitter[4].split_fraction[:, "outlet_1"].fix(0.9019)
    m.fs.t_splitter[5].split_fraction[:, "outlet_2"].fix(0.0381443)
    m.fs.t_splitter[6].split_fraction[:, "outlet_2"].fix(0.017535)
    m.fs.t_splitter[7].split_fraction[:, "outlet_2"].fix(0.0154)
    m.fs.t_splitter[8].split_fraction[:, "outlet_2"].fix(0.00121)

    # Fix the boiler inlet state. Note flow_mol is a dof and is already fixed
    m.fs.boiler.inlet.pressure.fix(boiler_inlet_state["pressure"])
    m.fs.boiler.inlet.enth_mol.fix(boiler_inlet_state["enthalpy"])

    # Fix the condenser mixer's drain's state
    m.fs.condenser_mix.drain.flow_mol.fix(condenser_drain_state["flow"])
    m.fs.condenser_mix.drain.pressure.fix(condenser_drain_state["pressure"])
    m.fs.condenser_mix.drain.enth_mol.fix(condenser_drain_state["enthalpy"])

    # Fix the state of drain for all mixers
    for i in m.fs.set_fwh_mix:
        m.fs.fwh_mix[i].drain.flow_mol.fix(fwh_mix_drain_state[i]["flow"])
        m.fs.fwh_mix[i].drain.pressure.fix(fwh_mix_drain_state[i]["pressure"])
        m.fs.fwh_mix[i].drain.enth_mol.fix(fwh_mix_drain_state[i]["enthalpy"])

    # During initialization, we deactivate the constraint that the shell_outlet
    # should be a saturated liquid. This is because, we are fixing the split
    # fraction, so the chosen flowrate may not meet the requirement. We will
    # activate the constraint later after all the units have been initialized.
    for i in m.fs.set_fwh:
        m.fs.fwh[i].vapor_frac_constraint.deactivate()

    solver = get_solver(options={"tol": 1e-6,
                                 "max_iter": 300,
                                 "halt_on_ampl_error": "yes"})

    # Initializing the boiler
    m.fs.boiler.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize HP splitter
    propagate_state(m.fs.boiler_to_hp_splitter)
    m.fs.hp_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the first turbine
    propagate_state(m.fs.hp_splitter_to_turbine_1)
    m.fs.turbine[1].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the first splitter
    propagate_state(m.fs.turbine_1_to_splitter_1)
    m.fs.t_splitter[1].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the second turbine
    propagate_state(m.fs.splitter_1_to_turbine_2)
    m.fs.turbine[2].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the second splitter
    propagate_state(m.fs.turbine_2_to_splitter_2)
    m.fs.t_splitter[2].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the reheater
    propagate_state(m.fs.splitter_2_to_reheater)
    m.fs.reheater.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize remaining turbines and splitters
    propagate_state(m.fs.reheater_to_turbine_3)
    for i in range(3, 10):
        m.fs.turbine[i].initialize(outlvl=outlvl, optarg=solver.options)

        if i < 9:  # Since there are only 8 splitters
            propagate_state(eval("m.fs.turbine_" + str(i) + "_to_splitter_" + str(i)))
            m.fs.t_splitter[i].initialize(outlvl=outlvl, optarg=solver.options)

            propagate_state(eval("m.fs.splitter_" + str(i) + "_to_turbine_" + str(i + 1)))

    # Initialize the boiler feed water turbine (Fixing the outlet pressure
    # to make it a square problem for initialization)
    propagate_state(m.fs.splitter_4_to_bfpt)
    m.fs.bfpt.outlet.pressure.fix(101325)
    m.fs.bfpt.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.bfpt.outlet.pressure.unfix()

    # Initialize the condenser mixer
    propagate_state(m.fs.bfpt_to_condenser_mix)
    propagate_state(m.fs.turbine_9_to_condenser_mix)
    m.fs.condenser_mix.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the condenser. We are fixing the molar flowrate of tube_inlet
    # to get a good initialization for feed water heaters
    propagate_state(m.fs.condenser_mix_to_condenser)
    m.fs.condenser.tube_inlet.flow_mol.fix(800000)
    m.fs.condenser.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.condenser.tube_inlet.flow_mol.unfix()

    # Initialize the condenser pump
    propagate_state(m.fs.condenser_to_pump)
    m.fs.cond_pump.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize feed water heaters, boiler feed water pump and mixers
    # associated with feed water heaters.
    # feed water heater 1
    propagate_state(m.fs.splitter_8_to_fwh_mix_1)
    m.fs.fwh_mix[1].initialize(outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.fwh_mix_1_to_fwh_1)
    propagate_state(m.fs.cond_pump_to_fwh_1)
    m.fs.fwh[1].initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 2 (Note: Drain state is fixed earlier)
    propagate_state(m.fs.splitter_7_to_fwh_mix_2)
    m.fs.fwh_mix[2].initialize(outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.fwh_mix_2_to_fwh_2)
    propagate_state(m.fs.fwh_1_to_fwh_2)
    m.fs.fwh[2].initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 3
    propagate_state(m.fs.splitter_6_to_fwh_mix_3)
    m.fs.fwh_mix[3].initialize(outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.fwh_mix_3_to_fwh_3)
    propagate_state(m.fs.fwh_2_to_fwh_3)
    m.fs.fwh[3].initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 4
    propagate_state(m.fs.splitter_5_to_fwh_4)
    propagate_state(m.fs.fwh_3_to_fwh_4)
    m.fs.fwh[4].initialize(outlvl=outlvl, optarg=solver.options)

    # Deaerator
    propagate_state(m.fs.splitter_4_to_fwh_mix_5)
    propagate_state(m.fs.fwh_4_to_fwh_mix_5)
    m.fs.fwh_mix[5].initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize bfp_splitter
    propagate_state(m.fs.fwh_mix_5_to_bfp_splitter)
    m.fs.bfp_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the TES system
    if m.fs.include_concrete_tes:
        propagate_state(m.fs.hp_splitter_to_tes)
        m.fs.tes.inlet_discharge.flow_mol.fix(discharge_flow)
        m.fs.tes.initialize(outlvl=outlvl, optarg=solver.options)

        propagate_state(m.fs.tes_to_fwh_mix_7)
        propagate_state(m.fs.tes_to_dis_turbine)
        m.fs.discharge_turbine.initialize(outlvl=outlvl, optarg=solver.options)

    # Boiler feed pump
    propagate_state(m.fs.bfp_splitter_to_bfp)
    m.fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 6
    propagate_state(m.fs.splitter_3_to_fwh_mix_6)
    m.fs.fwh_mix[6].initialize(outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.fwh_mix_6_to_fwh_6)
    propagate_state(m.fs.bfp_to_fwh_6)
    m.fs.fwh[6].initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 7
    propagate_state(m.fs.splitter_2_to_fwh_mix_7)
    m.fs.fwh_mix[7].initialize(outlvl=outlvl, optarg=solver.options)

    propagate_state(m.fs.fwh_mix_7_to_fwh_7)
    propagate_state(m.fs.fwh_6_to_fwh_7)
    m.fs.fwh[7].initialize(outlvl=outlvl, optarg=solver.options)

    # feed water heater 8
    propagate_state(m.fs.splitter_1_to_fwh_8)
    propagate_state(m.fs.fwh_7_to_fwh_8)
    m.fs.fwh[8].initialize(outlvl=outlvl, optarg=solver.options)

    # Unfix the values we fixed earlier for initialization
    # Unfix the inlet pressure and enthalpy of boiler. Flow_mol must remain fixed
    m.fs.boiler.inlet.pressure.unfix()
    m.fs.boiler.inlet.enth_mol.unfix()

    # Unfix the condenser mixer's drain's state
    m.fs.condenser_mix.drain.unfix()

    # Unfix the drain state of all the feed water heater mixers
    for i in m.fs.set_fwh_mix:
        m.fs.fwh_mix[i].drain.unfix()

    #  Unfix split fractions and activate vapor fraction constraints
    for i in m.fs.set_t_splitters:
        if i == 4:  # Split fraction to deaerator is fixed
            m.fs.t_splitter[i].split_fraction[:, "outlet_1"].unfix()
        else:
            m.fs.t_splitter[i].split_fraction[:, "outlet_2"].unfix()

        if i != 5:  # fwh[5] does not exist!
            m.fs.fwh[i].vapor_frac_constraint.activate()

    # for i in m.fs.set_turbines:
    #     m.fs.turbine[i].inlet.unfix()

    print(m.name, " Degrees of Freedom: ", degrees_of_freedom(m.fs))
    assert degrees_of_freedom(m.fs) == 0

    if outlvl == idaeslog.WARNING:
        res = solver.solve(m, tee=False)
    else:
        res = solver.solve(m, tee=True)
    
    print("Model Initialization = ", res.solver.termination_condition)

    print("*********************Model Initialized**************************")


def set_scaling_factors(m):
    """
    This function sets the missing scaling factors and any other scaling
    factors we want to include.
    """
    # for i in m.fs.set_turbines:
    #     iscale.set_scaling_factor(m.fs.turbine[i].control_volume.work, 1e-7)
    #
    # iscale.set_scaling_factor(m.fs.bfpt.control_volume.work, 1e-7)
    # iscale.set_scaling_factor(m.fs.boiler.control_volume.heat, 1e-9)
    # iscale.set_scaling_factor(m.fs.reheater.control_volume.heat, 1e-8)
    #
    # for i in m.fs.set_fwh:
    #     iscale.set_scaling_factor(m.fs.fwh[i].area, 1e-2)
    #     iscale.set_scaling_factor(m.fs.fwh[i].overall_heat_transfer_coefficient, 1e-3)
    #     iscale.set_scaling_factor(m.fs.fwh[i].shell.heat, 1e-7)
    #     iscale.set_scaling_factor(m.fs.fwh[i].tube.heat, 1e-7)
    #
    # iscale.set_scaling_factor(m.fs.condenser.shell.heat, 1e-9)
    # iscale.set_scaling_factor(m.fs.condenser.tube.heat, 1e-9)
    # iscale.set_scaling_factor(m.fs.cond_pump.control_volume.work, 1e-5)
    # iscale.set_scaling_factor(m.fs.bfp.control_volume.work, 1e-7)
    #
    # if m.fs.include_concrete_tes:
    #     m.fs.tes.set_default_scaling_factors()
    #     iscale.set_scaling_factor(m.fs.discharge_turbine.control_volume.work, 1)

    for i in m.fs.set_turbines:
        iscale.set_scaling_factor(m.fs.turbine[i].control_volume.work, 1e-6)

    iscale.set_scaling_factor(m.fs.bfpt.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.boiler.control_volume.heat, 1e-7)
    iscale.set_scaling_factor(m.fs.reheater.control_volume.heat, 1e-7)

    for i in m.fs.set_fwh:
        iscale.set_scaling_factor(m.fs.fwh[i].area, 1e-2)
        iscale.set_scaling_factor(m.fs.fwh[i].overall_heat_transfer_coefficient, 1e-3)
        iscale.set_scaling_factor(m.fs.fwh[i].shell.heat, 1e-7)
        iscale.set_scaling_factor(m.fs.fwh[i].tube.heat, 1e-7)

    iscale.set_scaling_factor(m.fs.condenser.shell.heat, 1e-7)
    iscale.set_scaling_factor(m.fs.condenser.tube.heat, 1e-7)
    iscale.set_scaling_factor(m.fs.cond_pump.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.bfp.control_volume.work, 1e-6)
    iscale.set_scaling_factor(m.fs.net_power_output, 1e-6)

    if m.fs.include_concrete_tes:
        m.fs.tes.set_default_scaling_factors()
        iscale.set_scaling_factor(m.fs.discharge_turbine.control_volume.work, 1e-4)

    iscale.calculate_scaling_factors(m)

    sf_enth_mol = m.fs.prop_water_mix.default_scaling_factor["enth_mol", None]
    sf_pressure = m.fs.prop_water_mix.default_scaling_factor["pressure", None]

    for i in m.fs.set_fwh:
        iscale.constraint_scaling_transform(
            m.fs.fwh[i].vapor_frac_constraint[0], sf_enth_mol)
        iscale.constraint_scaling_transform(
            m.fs.fwh[i].s1_pdrop_constraint[0], sf_pressure)
        iscale.constraint_scaling_transform(
            m.fs.fwh[i].s2_pdrop_constraint[0], sf_pressure)

    for i in m.fs.set_fwh_mix:
        iscale.constraint_scaling_transform(
            m.fs.fwh_mix[i].mixer_pressure_constraint[0], sf_pressure)

    iscale.constraint_scaling_transform(
        m.fs.condenser_mix.mixer_pressure_constraint[0], sf_pressure)
    iscale.constraint_scaling_transform(
        m.fs.constraint_bfpt_out_pressure[0], sf_pressure)
    iscale.constraint_scaling_transform(
        m.fs.constraint_bfp_power[0], 1e-6)
    iscale.constraint_scaling_transform(
        m.fs.production_cons[0], 1e-6)

    if m.fs.include_concrete_tes:
        iscale.constraint_scaling_transform(
            m.fs.discharge_turbine.turbine_enthalpy_constraint[0], sf_pressure)

        for t, con in m.fs.tes.charge_inlet_enth_mol_equality.items():
            iscale.constraint_scaling_transform(con, sf_enth_mol)

        for t, con in m.fs.tes.charge_inlet_pressure_equality.items():
            iscale.constraint_scaling_transform(con, sf_pressure)

        for t, con in m.fs.tes.charge_outlet_enth_mol_equality.items():
            iscale.constraint_scaling_transform(con, sf_enth_mol)

        for t, con in m.fs.tes.charge_outlet_pressure_equality.items():
            iscale.constraint_scaling_transform(con, sf_pressure)

        for t, con in m.fs.tes.discharge_inlet_enth_mol_equality.items():
            iscale.constraint_scaling_transform(con, sf_enth_mol)

        for t, con in m.fs.tes.discharge_inlet_pressure_equality.items():
            iscale.constraint_scaling_transform(con, sf_pressure)

        for t, con in m.fs.tes.discharge_outlet_enth_mol_equality.items():
            iscale.constraint_scaling_transform(con, sf_enth_mol)

        for t, con in m.fs.tes.discharge_outlet_pressure_equality.items():
            iscale.constraint_scaling_transform(con, sf_pressure)


def unfix_dof_for_optimization(m):
    """
    This function unfixes a few degrees of freedom for optimization
    For now, we will not unfix the design of the tes system. Also,
    we will not unfix the water flowrate to the boiler.
    Therefore, the boiler will always be maintained at its base load.
    The only decision we need to make is whether to produce power and
    sell it to the grid, and/or charge/discharge the TES. Since we have
    a separate discharge turbine, the total power produced could exceed
    the maximum rated capacity of the power plant.
    """
    if m.fs.include_concrete_tes:
        # Unfix the split fraction of the hp_splitter and specify bounds
        m.fs.hp_splitter.split_fraction[:, "outlet_2"].unfix()
        m.fs.hp_splitter.split_fraction[:, "outlet_2"].setlb(1e-5)
        m.fs.hp_splitter.split_fraction[:, "outlet_2"].setub(0.15)

        # Fixme: Currently, we have a separate inlet for the discharge.
        # so, the split fraction of bfp_splitter must be fixed. If this changed,
        # the the following lines need to be uncommented.
        # Unfix the split fraction of the bfp_spiltter
        # m.fs.bfp_splitter.split_fraction[:, "outlet_2"].unfix()
        # m.fs.bfp_splitter.split_fraction[:, "outlet_2"].setlb(1e-6)
        # m.fs.bfp_splitter.split_fraction[:, "outlet_2"].setub(0.05)

        m.fs.tes.inlet_discharge.flow_mol.unfix()
        m.fs.tes.inlet_discharge.flow_mol.setlb(1)
        m.fs.tes.inlet_discharge.flow_mol.setub(1650)

        # Unfix the initial temperature profile of the concrete block
        # Bounds on temperatures are set in the unit model
        m.fs.tes.period[1].concrete.init_temperature.unfix()

    else:
        # Unfix the boiler feed water flow
        m.fs.boiler.inlet.flow_mol.unfix()
        m.fs.boiler.inlet.flow_mol.setlb(14000)
        m.fs.boiler.inlet.flow_mol.setub(29112)


def generate_stream_table(m):
    blk = ConcreteModel()
    blk.thermo_params = iapws95.Iapws95ParameterBlock()
    blk.state = blk.thermo_params.build_state_block(
        defined_state=True,
        has_phase_equilibrium=True,
    )
    blk.state.flow_mol.fix(1)

    zz = {}
    for v in m.component_objects(Arc):
        zz[v.name] = {"flow_mol": v.ports[0].flow_mol[0].value,
                      "enth_mol": v.ports[0].enth_mol[0].value,
                      "pressure": v.ports[0].pressure[0].value}
        blk.state.pressure.fix(v.ports[0].pressure[0].value)
        blk.state.enth_mol.fix(v.ports[0].enth_mol[0].value)
        zz[v.name]["temperature"] = blk.state.temperature.expr()

    df = pd.DataFrame(zz)
    df.to_excel("SCPC_stream_table.xlsx")
