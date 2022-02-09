__author__ = "Andres J Calderon, Naresh Susarla, Miguel Zamarripa, Jaffer Ghouse"

# Import Pyomo libraries
from pyomo.environ import (ConcreteModel,
                           Var,
                           RangeSet,
                           ConstraintList,
                           TransformationFactory)
from pyomo.network import Arc

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.generic_models.unit_models import (HeatExchanger,
                                              PressureChanger,
                                              MomentumMixingType,
                                              Heater,
                                              Separator)

from idaes.power_generation.unit_models.helm import (HelmSplitter,
                                                     HelmTurbineStage,
                                                     HelmMixer as Mixer,
                                                     HelmIsentropicCompressor as WaterPump,
                                                     HelmNtuCondenser as CondenserHelm)

from idaes.generic_models.unit_models.heat_exchanger import delta_temperature_underwood_callback
from tube_concrete_model import ConcreteTES
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.generic_models.properties import iapws95

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
"""
List of changes:
Currently we are considering two possible locations for steam withdrawal:
either directly from boiler or from reheater. In either case, the pressure of the 
water at the outlet of the TES will be high enough to be put directly in 
fwh_mix_7. Therefore, the bfp_mix model is not needed, so I'm removing the model
to avoid confusion. In future, suppose we want to include the possibility to
withdraw steam from turbine[4]. Even in this case, we do not need the bfp_mix model.
We can put the water directly in Deaerator. 
"""
# TODO: Do we need storage cooler, hx_pump? Probably not if we are going to
#       extract the steam only from HP splitter.
#       Is it a good idea to extract steam from IP splitter for charging?


def build_scpp_flowsheet(m):
    """
    This function builds an instance of the SCPP flowsheet
    """
    print(m.name)
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Add thermodynamic packages
    m.fs.prop_water_lg = iapws95.Iapws95ParameterBlock(
        default={"phase_presentation": iapws95.PhaseType.LG})
    m.fs.prop_water_mix = iapws95.Iapws95ParameterBlock()

    """
    TURBINE TRAIN: This part builds, turbine models, splitters models
    after each turbine, hp_splitter and ip_splitter models
    """
    # Add HP steam splitter
    m.fs.hp_splitter = HelmSplitter(default={
        "property_package": m.fs.prop_water_mix})

    # Add IP steam splitter
    m.fs.ip_splitter = HelmSplitter(default={
        "property_package": m.fs.prop_water_mix})

    # Add turbine stages
    m.fs.turbine = HelmTurbineStage(RangeSet(9), default={
        "property_package": m.fs.prop_water_mix})

    # Add boiler feed water turbine
    m.fs.bfpt = HelmTurbineStage(default={
        "property_package": m.fs.prop_water_mix})

    # Add splitters associated with turbines. Except splitter 4,
    # all splitters have 2 outlets. Splitter 4 has three outlets
    m.fs.t_splitter = HelmSplitter(RangeSet(8),
                                   default={"property_package": m.fs.prop_water_lg,
                                            "num_outlets": 2},
                                   # Set the number of outlets for splitter 4
                                   initialize={4: {"property_package": m.fs.prop_water_lg,
                                                   "num_outlets": 3}})

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
    m.fs.boiler = Heater(default={
        "property_package": m.fs.prop_water_lg,
        "has_pressure_change": True})

    # Add reheater
    m.fs.reheater = Heater(default={
        "property_package": m.fs.prop_water_lg,
        "has_pressure_change": True})

    # Outlet temperature of boiler is set to 866.15 K
    @m.fs.boiler.Constraint(m.fs.time)
    def boiler_temperature_constraint(blk, t):
        return blk.control_volume.properties_out[t].temperature == 866.15  # K

    # Outlet temperature of reheater is set to 866.15 K
    @m.fs.reheater.Constraint(m.fs.time)
    def reheater_temperature_constraint(blk, t):
        return blk.control_volume.properties_out[t].temperature == 866.15  # K

    # Add feed water heaters
    fwh_list = [1, 2, 3, 4, 6, 7, 8]
    m.fs.fwh = HeatExchanger(fwh_list, default={
        "delta_temperature_callback": delta_temperature_underwood_callback,
        "shell": {"property_package": m.fs.prop_water_lg,
                  "material_balance_type": MaterialBalanceType.componentTotal,
                  "has_pressure_change": True},
        "tube": {"property_package": m.fs.prop_water_lg,
                 "material_balance_type": MaterialBalanceType.componentTotal,
                 "has_pressure_change": True}})

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
    for i in fwh_list:
        obj = m.fs.fwh[i]

        # Outlet enthalpy of condensate in an FWH must be same as that of saturated liquid
        @obj.Constraint(m.fs.time)
        def vapor_frac_constraint(blk, t):
            return (blk.side_1.properties_out[t].enth_mol ==
                    blk.side_1.properties_out[t].enth_mol_sat_phase['Liq'])

        # Setting a 4% pressure drop on the feedwater side (P_out = 0.96 * P_in)
        @obj.Constraint(m.fs.time)
        def s2_pdrop_constraint(blk, t):
            return (blk.side_2.properties_out[t].pressure ==
                    0.96 * blk.side_2.properties_in[t].pressure)

        # Setting the outlet pressure of condensate to be 10% more than that of
        # steam routed to condenser, as described in FWH description
        @obj.Constraint(m.fs.time)
        def s1_pdrop_constraint(blk, t):
            return (blk.side_1.properties_out[t].pressure ==
                    1.1 * pressure_ratio_list[i] * blk.side_1.properties_in[t].pressure)

    """
    FEED WATER MIXERS: This part adds mixers associated with feed water heaters
    """
    # Mixer 5 has an additional inlet from feedwater heater, and Mixer 7 has an
    # additional inlet from hx_pump
    fwh_mix_list = [1, 2, 3, 5, 6, 7]
    m.fs.fwh_mix = Mixer(fwh_mix_list,
                         default={
                             "momentum_mixing_type": MomentumMixingType.none,
                             "inlet_list": ["steam", "drain"],
                             "property_package": m.fs.prop_water_lg},
                         initialize={
                             # Add feedwater inlet to the deaerator
                             5: {"momentum_mixing_type": MomentumMixingType.none,
                                 "inlet_list": ["steam", "drain", "feedwater"],
                                 "property_package": m.fs.prop_water_lg},
                             # Add an inlet from hx_pump to mixer 7
                             7: {"momentum_mixing_type": MomentumMixingType.none,
                                 "inlet_list": ["steam", "drain", "from_hx_pump"],
                                 "property_package": m.fs.prop_water_lg}})

    # Pressure constraint for all mixers
    for i in fwh_mix_list:
        obj = m.fs.fwh_mix[i]

        if i == 5:  # Deaerator
            # Since the deaerator's (fwh_mix[5]) 'feedwater' inlet has the
            # minimum pressure, the outlet pressure must be the same as that
            # of the 'feedwater' inlet

            @obj.Constraint(m.fs.time)
            def mixer_pressure_constraint(blk, t):
                return blk.feedwater_state[t].pressure == blk.mixed_state[t].pressure

        else:
            # For the remaining mixers, the steam from the splitters has the minimum
            # pressure, so the outlet pressure must be the same as that of inlet steam

            @obj.Constraint(m.fs.time)
            def mixer_pressure_constraint(blk, t):
                return blk.steam_state[t].pressure == blk.mixed_state[t].pressure

    """
    CONDENSATION STAGE: This part adds models for all the unit operations 
    related to the heat removal process.
    """
    # Add condenser mixer
    # The inlet 'main' refers to the main steam coming from the turbine train
    # Inlet 'bfpt' refers to the steam coming from the bolier feed pump turbine
    # Inlet 'drain' refers to the condensed steam from the feed water heater 1
    # Inlet 'makeup' refers to the make up water
    m.fs.condenser_mix = Mixer(default={
        "momentum_mixing_type": MomentumMixingType.none,
        "inlet_list": ["main", "bfpt", "drain", "makeup"],
        "property_package": m.fs.prop_water_lg})

    # The outlet pressure of condenser mixer is equal to the minimum pressure
    # Since the turbine (#9) outlet (or, mixer inlet 'main') pressure
    # has the minimum pressure, the following constraint sets the outlet
    # pressure of the condenser mixer to the pressure of the inlet 'main'
    @m.fs.condenser_mix.Constraint(m.fs.time)
    def mixer_pressure_constraint(blk, t):
        return blk.main_state[t].pressure == blk.mixed_state[t].pressure

    # Add condenser
    m.fs.condenser = CondenserHelm(default={
        "shell": {"has_pressure_change": False,
                  "property_package": m.fs.prop_water_mix},
        "tube": {"has_pressure_change": False,
                 "property_package": m.fs.prop_water_mix}})

    """
    PUMPS: This part adds models for all the pumps used in the flowsheet
    """
    # Add condensate pump (Pumps water from the condenser outlet)
    m.fs.cond_pump = WaterPump(default={
        "property_package": m.fs.prop_water_mix})

    # Add boiler feed water pump (Pumps water between feed water heaters)
    m.fs.bfp = WaterPump(default={
        "property_package": m.fs.prop_water_mix})

    # Add a splitter before bfp to extract water for the discharge
    m.fs.bfp_splitter = HelmSplitter(default={
        "property_package": m.fs.prop_water_mix})

    """
    STORAGE: This part adds TES model and the models for auxiliary
    equipment required for the TES unit
    """
    # Todo: Pass this information as input
    # Add Concrete TES model
    data1 = {
        "delta_time": 1800,
        "time_periods": 2,
        "segments": 20,
        "deltaP": 0,
        "concrete_init_temp": [750, 732.631579, 715.2631579, 697.8947368, 680.5263158, 663.1578947,
                               645.7894737, 628.4210526, 611.0526316, 593.6842105, 576.3157895, 558.9473684,
                               541.5789474, 524.2105263, 506.8421053, 489.4736842, 472.1052632, 454.7368421,
                               437.3684211, 420],
        "tube_length": 64.9,
        "tube_diameter": 0.0105664,
        "number_tubes": 10000,
        "concrete_area": 0.00847,
        "concrete_conductivity": 1,
        "concrete_density": 2240,
        "concrete_specific_heat": 900,
        # This data is used for initialization, concrete_final_temp is recalculated during the initialization step
        "concrete_final_temp": [787.049044, 768.2461577, 749.7581953, 731.6762425, 713.9605891, 696.5867905,
                                679.5450987, 662.8403567, 646.4919514, 630.5145062, 614.9422052, 599.8251406,
                                585.2301277, 571.242303, 557.9698795, 545.6035881, 534.0878954, 523.0092146,
                                511.8313733, 500.2123965],
        "flow_mol": 0.00958 * 1000 / 18.01528,
        "inlet_pressure": 19600000,
        "inlet_temperature": 853.92699435}

    m.fs.tes = ConcreteTES(default={
        "model_data": data1,
        "property_package": m.fs.prop_water_lg})

    # Connect the TES block with hp splitter and storage cooler
    m.fs.tes_connections = ConstraintList()

    # Molar flow, pressure and enthalpy at inlet
    # Fixme: Use Arcs, and just modify the flow_mol constraint
    for p in m.fs.tes.time_periods:
        m.fs.tes_connections.add(
            expr=m.fs.hp_splitter.outlet_2.flow_mol[0] ==
                 data1['number_tubes'] * m.fs.tes.period[p].tube_charge.tube_inlet.flow_mol[0])
        m.fs.tes_connections.add(
            expr=m.fs.hp_splitter.outlet_2.pressure[0] ==
                 m.fs.tes.period[p].tube_charge.tube_inlet.pressure[0])
        m.fs.tes_connections.add(
            expr=m.fs.hp_splitter.outlet_2.enth_mol[0] ==
                 m.fs.tes.period[p].tube_charge.tube_inlet.enth_mol[0])

    # This heat exchanger cools the water leaving the concrete TES unit
    m.fs.storage_cooler = Heater(default={
        "property_package": m.fs.prop_water_mix,
        "has_pressure_change": True})

    # The outlet stream from the cooler is required to be subcooled, i.e., its enthalpy
    # must be below the enthalpy of saturation. This condition was selected instead of
    # temperatures, because it helps with the convergence of the model.
    # return (m.fs.storage_cooler.control_volume.properties_out[0].temperature <=
    #             m.fs.storage_cooler.control_volume.properties_out[0].temperature_sat - 5)
    @m.fs.storage_cooler.Constraint(m.fs.time)
    def cooler_enthalpy_constraint(blk, t):
        return (blk.control_volume.properties_out[t].enth_mol ==
                blk.control_volume.properties_out[t].enth_mol_sat_phase['Liq'])

    p = data1['time_periods']
    m.fs.tes_connections.add(
        expr=data1['number_tubes'] * m.fs.tes.period[p].tube_charge.tube_outlet.flow_mol[0] ==
             m.fs.storage_cooler.inlet.flow_mol[0])
    m.fs.tes_connections.add(
        expr=m.fs.tes.period[p].tube_charge.tube_outlet.pressure[0] ==
             m.fs.storage_cooler.inlet.pressure[0])
    m.fs.tes_connections.add(
        expr=m.fs.tes.period[p].tube_charge.tube_outlet.enth_mol[0] ==
             m.fs.storage_cooler.inlet.enth_mol[0])

    # Connect the outlet_2 of bfp_splitter to the cold side of tes
    for p in m.fs.tes.time_periods:
        m.fs.tes_connections.add(
            expr=m.fs.bfp_splitter.outlet_2.flow_mol[0] ==
                 data1['number_tubes'] * m.fs.tes.period[p].tube_discharge.tube_inlet.flow_mol[0])
        m.fs.tes_connections.add(
            expr=m.fs.bfp_splitter.outlet_2.pressure[0] ==
                 m.fs.tes.period[p].tube_discharge.tube_inlet.pressure[0])
        m.fs.tes_connections.add(
            expr=m.fs.bfp_splitter.outlet_2.enth_mol[0] ==
                 m.fs.tes.period[p].tube_discharge.tube_inlet.enth_mol[0])

    # Pump to increase the pressure of the water leaving the TES unit
    m.fs.hx_pump = WaterPump(default={
        "property_package": m.fs.prop_water_mix})

    """
    DISCHARGE TRAIN: This part adds the turbine for the discharge 
    process, and other associated equipment.
    """
    m.fs.discharge_turbine = HelmTurbineStage(default={
        "property_package": m.fs.prop_water_mix})

    # To avoid condensation inside the turbine, we impose the following
    # constraint on the outlet
    @m.fs.discharge_turbine.Constraint(m.fs.time)
    def turbine_enthalpy_constraint(blk, t):
        # return (blk.control_volume.properties_out[t].enth_mol ==
        #         blk.control_volume.properties_out[t].enth_mol_sat_phase['Vap'])
        # return blk.control_volume.properties_out[t].temperature == 310
        return blk.control_volume.properties_out[t].pressure == 6644

    # Connect the outlet of tube_discharge to discharge_turbine
    p = data1['time_periods']
    m.fs.tes_connections.add(
        expr=data1['number_tubes'] * m.fs.tes.period[p].tube_discharge.tube_outlet.flow_mol[0] ==
             m.fs.discharge_turbine.inlet.flow_mol[0])
    m.fs.tes_connections.add(
        expr=m.fs.tes.period[p].tube_discharge.tube_outlet.pressure[0] ==
             m.fs.discharge_turbine.inlet.pressure[0])
    m.fs.tes_connections.add(
        expr=m.fs.tes.period[p].tube_discharge.tube_outlet.enth_mol[0] ==
             m.fs.discharge_turbine.inlet.enth_mol[0])

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
    m.fs.net_power_output = Var(m.fs.time,
                                initialize=620 * 1e6,
                                doc="Net Power We out from the power plant")

    #   Constraint on Plant Power Output
    #   Plant Power Out = Turbine Power - Power required for HX Pump
    @m.fs.Constraint(m.fs.time)
    def production_cons(blk, t):
        return (sum(blk.turbine[j].work_mechanical[t] for j in RangeSet(9))
                + m.fs.cond_pump.control_volume.work[t] == -m.fs.net_power_output[t])

    create_arcs(m)

    return m


def create_arcs(m):
    """
    TURBINE TRAIN: This part connects models in turbine train
    """
    splitter_list = RangeSet(8)

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

    for i in splitter_list:
        # Add arc connecting turbine and splitter
        setattr(m.fs, "turbine_" + str(i) + "_to_splitter_" + str(i),
                Arc(source=m.fs.turbine[i].outlet,
                    destination=m.fs.t_splitter[i].inlet))

        # Add arc connecting the splitter's outlet_1 to turbine/reheater
        if i == 2:
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
                        destination=m.fs.fwh[split_fwh_map[i][1]].inlet_1))
        else:
            setattr(m.fs, "splitter_" + str(i) + "_to_fwh_mix_" + str(split_fwh_map[i][1]),
                    Arc(source=m.fs.t_splitter[i].outlet_2,
                        destination=m.fs.fwh_mix[split_fwh_map[i][1]].steam))

    m.fs.splitter_4_to_bfpt = Arc(source=m.fs.t_splitter[4].outlet_3,
                                  destination=m.fs.bfpt.inlet)

    """
    BOILER: This part connects models related to boilers/feed water heaters
    """
    m.fs.boiler_to_hp_splitter = Arc(source=m.fs.boiler.outlet,
                                     destination=m.fs.hp_splitter.inlet)

    m.fs.hp_splitter_to_turbine_1 = Arc(source=m.fs.hp_splitter.outlet_1,
                                        destination=m.fs.turbine[1].inlet)

    m.fs.reheater_to_ip_splitter = Arc(source=m.fs.reheater.outlet,
                                       destination=m.fs.ip_splitter.inlet)

    m.fs.ip_splitter_to_turbine_3 = Arc(source=m.fs.ip_splitter.outlet_1,
                                        destination=m.fs.turbine[3].inlet)

    for i in [1, 2, 3, 5, 6, 7]:  # List of fwh mixers
        # Add arcs to connect the outlet of fwh_mix to fwh/bfp
        if i == 5:
            # Outlet of Deaerator goes to bfp_splitter
            m.fs.fwh_mix_5_to_bfp_splitter = \
                Arc(source=m.fs.fwh_mix[5].outlet,
                    destination=m.fs.bfp_splitter.inlet)

            # Outlet_1 of bfp_splitter goes to bfp
            m.fs.bfp_splitter_to_bfp = Arc(source=m.fs.bfp_splitter.outlet_1,
                                           destination=m.fs.bfp.inlet)

        else:
            # Outlet of each fwh_mix goes to the inlet_1 of fwh
            setattr(m.fs, "fwh_mix_" + str(i) + "_to_fwh_" + str(i),
                    Arc(source=m.fs.fwh_mix[i].outlet,
                        destination=m.fs.fwh[i].inlet_1))

        # Add arcs connecting the drains and the mixers
        setattr(m.fs, "fwh_" + str(i + 1) + "_to_fwh_mix_" + str(i),
                Arc(source=m.fs.fwh[i + 1].outlet_1,
                    destination=m.fs.fwh_mix[i].drain))

        # Add arcs connecting feed water outlet to feed water inlet
        if i != 5:  # Skip Deaerator
            setattr(m.fs, "fwh_" + str(i) + "_to_fwh_" + str(i + 1),
                    Arc(source=m.fs.fwh[i].outlet_2,
                        destination=m.fs.fwh[i + 1].inlet_2))

    m.fs.fwh_4_to_fwh_mix_5 = Arc(source=m.fs.fwh[4].outlet_2,
                                  destination=m.fs.fwh_mix[5].feedwater)

    m.fs.fwh_8_to_boiler = Arc(source=m.fs.fwh[8].outlet_2,
                               destination=m.fs.boiler.inlet)

    """
    CONDENSER: This part connects models related to the heat removal process
    """
    m.fs.cooler_to_hx_pump = Arc(source=m.fs.storage_cooler.outlet,
                                 destination=m.fs.hx_pump.inlet)

    # Last turbine outlet to the mixer associated with the condenser
    m.fs.turbine_9_to_condenser_mix = Arc(source=m.fs.turbine[9].outlet,
                                          destination=m.fs.condenser_mix.main)

    # FWH to the mixer associated with condenser
    m.fs.fwh_1_to_condenser_mix = Arc(source=m.fs.fwh[1].outlet_1,
                                      destination=m.fs.condenser_mix.drain)

    # Boiler feed water turbine to the mixer associated with condenser
    m.fs.bfpt_to_condenser_mix = Arc(source=m.fs.bfpt.outlet,
                                     destination=m.fs.condenser_mix.bfpt)

    # Mixer associated with condenser to condenser
    m.fs.condenser_mix_to_condenser = Arc(source=m.fs.condenser_mix.outlet,
                                          destination=m.fs.condenser.inlet_1)

    # Condenser outlet to condenser pump
    m.fs.condenser_to_pump = Arc(source=m.fs.condenser.outlet_1,
                                 destination=m.fs.cond_pump.inlet)

    """
    PUMPS AND CONCRETE TES CONNECTIONS
    """
    # Condenser pump outlet to feed water heater
    m.fs.cond_pump_to_fwh_1 = Arc(source=m.fs.cond_pump.outlet,
                                  destination=m.fs.fwh[1].inlet_2)

    # Mix the water from the concrete TES unit
    m.fs.hx_pump_to_fwh_mix_7 = Arc(source=m.fs.hx_pump.outlet,
                                    destination=m.fs.fwh_mix[7].from_hx_pump)

    # Boiler feed water pump mixer to feed water heater
    m.fs.bfp_to_fwh_6 = Arc(source=m.fs.bfp.outlet,
                            destination=m.fs.fwh[6].inlet_2)

    TransformationFactory("network.expand_arcs").apply_to(m.fs)


def fix_dof_and_initialize(m, outlvl=idaeslog.INFO_HIGH):
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
    10. Condenser has four dof: pressure and molar enthalpy of inlet_2, area and
        heat transfer coefficient. The condenser model has a constraint that
        governs the molar flowrate of inlet_2.
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
    m.fs.hp_splitter.split_fraction[:, "outlet_2"].fix(0.1)
    m.fs.ip_splitter.split_fraction[:, "outlet_2"].fix(0.0)
    m.fs.t_splitter[4].split_fraction[:, "outlet_2"].fix(0.050331)
    m.fs.bfp_splitter.split_fraction[:, "outlet_2"].fix(0)

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
    m.fs.condenser.inlet_2.pressure.fix(500000)
    m.fs.condenser.inlet_2.enth_mol.fix(1800)
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

    # Heat exchanger pump
    m.fs.hx_pump.efficiency_isentropic.fix(0.80)
    m.fs.hx_pump.outlet.pressure.fix(main_steam_pressure)

    # Storage cooler
    m.fs.storage_cooler.deltaP.fix(0)

    # Fix degrees of freedom of the discharge turbine. we need to fix
    # only one dof for this turbine. The other dof is specified as a
    # constraint on the state of the outlet.
    m.fs.discharge_turbine.efficiency_isentropic.fix(0.75)

    """
    Initialization Begins
    """
    # Some numbers useful for initialization, but THESE ARE NOT DOF. We fix these
    # quantities at the beginning of the initialization process and unfix them
    # once the flowsheet has been initialized
    boiler_inlet_state = {"pressure": 24657896, "enthalpy": 20004}
    condenser_drain_state = {"flow": 1460, "pressure": 7308, "enthalpy": 2973}
    fwh_mix_drain_state = {1: {"flow": 1434, "pressure": 14617, "enthalpy": 3990},
                           2: {"flow": 1136, "pressure": 35685, "enthalpy": 5462},
                           3: {"flow": 788, "pressure": 87123, "enthalpy": 7160},
                           5: {"flow": 6207, "pressure": 519291, "enthalpy": 11526},
                           6: {"flow": 5299, "pressure": 2177587, "enthalpy": 16559},
                           7: {"flow": 3730, "pressure": 5590711, "enthalpy": 21232}}

    # Fix the boiler inlet state. Note flow_mol is a dof and is already fixed
    m.fs.boiler.inlet.pressure.fix(boiler_inlet_state["pressure"])
    m.fs.boiler.inlet.enth_mol.fix(boiler_inlet_state["enthalpy"])

    # Fix the condenser mixer's drain's state
    m.fs.condenser_mix.drain.flow_mol.fix(condenser_drain_state["flow"])
    m.fs.condenser_mix.drain.pressure.fix(condenser_drain_state["pressure"])
    m.fs.condenser_mix.drain.enth_mol.fix(condenser_drain_state["enthalpy"])

    # Fix the state of drain for all mixers
    for i in [1, 2, 3, 5, 6, 7]:
        m.fs.fwh_mix[i].drain.flow_mol.fix(fwh_mix_drain_state[i]["flow"])
        m.fs.fwh_mix[i].drain.pressure.fix(fwh_mix_drain_state[i]["pressure"])
        m.fs.fwh_mix[i].drain.enth_mol.fix(fwh_mix_drain_state[i]["enthalpy"])

    # Fixing the split fractions of the splitters. Will be unfixed later
    m.fs.t_splitter[1].split_fraction[:, "outlet_2"].fix(0.12812)
    m.fs.t_splitter[2].split_fraction[:, "outlet_2"].fix(0.061824)
    m.fs.t_splitter[3].split_fraction[:, "outlet_2"].fix(0.03815)
    m.fs.t_splitter[4].split_fraction[:, "outlet_1"].fix(0.9019)
    m.fs.t_splitter[5].split_fraction[:, "outlet_2"].fix(0.0381443)
    m.fs.t_splitter[6].split_fraction[:, "outlet_2"].fix(0.017535)
    m.fs.t_splitter[7].split_fraction[:, "outlet_2"].fix(0.0154)
    m.fs.t_splitter[8].split_fraction[:, "outlet_2"].fix(0.00121)

    # During initialization, we deactivate the constraint that the outlet_1
    # should be a saturated liquid. This is because, we are fixing the split
    # fraction, so the chosen flowrate may not meet the requirement. We will
    # activate the constraint later after all the units have been initialized.
    for i in [1, 2, 3, 4, 6, 7, 8]:
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

    # Initialize the IP splitter
    propagate_state(m.fs.reheater_to_ip_splitter)
    m.fs.ip_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize remaining turbines and splitters
    propagate_state(m.fs.ip_splitter_to_turbine_3)
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

    # Initialize the condenser. We are fixing the molar flowrate of inlet_2
    # to get a good initialization for feed water heaters
    propagate_state(m.fs.condenser_mix_to_condenser)
    m.fs.condenser.inlet_2.flow_mol.fix(800000)
    m.fs.condenser.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.condenser.inlet_2.flow_mol.unfix()

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

    # feed water heater 2
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
    # Todo: Need to model the problem such that we can use propagate_state
    for p in m.fs.tes.time_periods:
        m.fs.tes.period[p].tube_charge.tube_inlet.flow_mol[0].value = \
            (m.fs.hp_splitter.outlet_2.flow_mol[0].value /
             m.fs.tes.config.model_data["number_tubes"])
        m.fs.tes.period[p].tube_charge.tube_inlet.pressure[0].value = \
            m.fs.hp_splitter.outlet_2.pressure[0].value
        m.fs.tes.period[p].tube_charge.tube_inlet.enth_mol[0].value = \
            m.fs.hp_splitter.outlet_2.enth_mol[0].value

        m.fs.tes.period[p].tube_discharge.tube_inlet.flow_mol[0].value = \
            (m.fs.bfp_splitter.outlet_2.flow_mol[0].value /
             m.fs.tes.config.model_data["number_tubes"])
        m.fs.tes.period[p].tube_discharge.tube_inlet.pressure[0].value = \
            m.fs.bfp_splitter.outlet_2.pressure[0].value
        m.fs.tes.period[p].tube_discharge.tube_inlet.enth_mol[0].value = \
            m.fs.bfp_splitter.outlet_2.enth_mol[0].value

    m.fs.tes.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize the storage cooler
    p = m.fs.tes.config.model_data['time_periods']
    m.fs.storage_cooler.inlet.flow_mol[0].value = \
        (m.fs.tes.period[p].tube_charge.tube_outlet.flow_mol[0].value *
         m.fs.tes.config.model_data["number_tubes"])
    m.fs.storage_cooler.inlet.pressure[0].value = \
        m.fs.tes.period[p].tube_charge.tube_outlet.pressure[0].value
    m.fs.storage_cooler.inlet.enth_mol[0].value = \
        m.fs.tes.period[p].tube_charge.tube_outlet.enth_mol[0].value
    m.fs.storage_cooler.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize hx_pump
    propagate_state(m.fs.cooler_to_hx_pump)
    m.fs.hx_pump.initialize(outlvl=outlvl, optarg=solver.options)

    # Initialize discharge turbine
    p = m.fs.tes.config.model_data['time_periods']
    m.fs.discharge_turbine.inlet.flow_mol[0].value = \
        (m.fs.tes.period[p].tube_discharge.tube_outlet.flow_mol[0].value *
         m.fs.tes.config.model_data["number_tubes"])
    m.fs.discharge_turbine.inlet.pressure[0].value = \
        m.fs.tes.period[p].tube_discharge.tube_outlet.pressure[0].value
    m.fs.discharge_turbine.inlet.enth_mol[0].value = \
        m.fs.tes.period[p].tube_discharge.tube_outlet.enth_mol[0].value
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
    propagate_state(m.fs.hx_pump_to_fwh_mix_7)
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
    for i in [1, 2, 3, 5, 6, 7]:
        m.fs.fwh_mix[i].drain.unfix()

    #  Unfix split fractions and activate vapor fraction constraints
    for i in range(1, 9):
        if i == 4:  # Split fraction to deaerator is fixed
            m.fs.t_splitter[i].split_fraction[:, "outlet_1"].unfix()
        else:
            m.fs.t_splitter[i].split_fraction[:, "outlet_2"].unfix()

        if i != 5:  # fwh[5] does not exist!
            m.fs.fwh[i].vapor_frac_constraint.activate()

    print("Degrees of Freedom: ", degrees_of_freedom(m.fs))
    assert degrees_of_freedom(m.fs) == 0
    res = solver.solve(m, tee=True)
    print("Model Initialization = ", res.solver.termination_condition)
    print("*********************Model Initialized**************************")


def set_scaling_factors(m):
    pass


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
    # Unfix the split fraction of the hp_splitter
    m.fs.hp_splitter.split_fraction[:, "outlet_2"].unfix()

    # Unfix the split fraction of the bfp_spiltter
    m.fs.bfp_splitter.split_fraction[:, "outlet_2"].unfix()

    # Unfix the initial temperature profile of the concrete block
    m.fs.tes.period[1].concrete.init_temperature.unfix()


if __name__ == "__main__":
    mdl = ConcreteModel()
    build_scpp_flowsheet(mdl)
    fix_dof_and_initialize(mdl)

    print("End of the run!")
