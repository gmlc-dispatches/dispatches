##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program
# (DISPATCHES), and is copyright (c) 2021 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory, National Technology & Engineering Solutions of Sandia,
# LLC, Alliance for Sustainable Energy, LLC, Battelle Energy Alliance, LLC,
# University of Notre Dame du Lac, et al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information, respectively. Both files are also available online
# at the URL: "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################

"""
This is a simple model for an ultrasupercritical coal-fired power plant
integrated with a boiler fire-side unit model and a CO2 capture system.
The boiler fire-side model accounts for flue-gas generation.
The CO2 capture system uses surrogates for a solvent-based piperazine system.
The capture rate is fixed to 90%.
"""

from pyomo.network import Arc
from pyomo.environ import TransformationFactory, value
# IDAES Imports
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.power_generation.unit_models.boiler_fireside import BoilerFireside
from idaes.generic_models.unit_models import Heater, MomentumMixingType
from idaes.core.util import get_solver
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.power_generation.unit_models.helm import HelmMixer, HelmSplitter
from idaes.power_generation.properties import FlueGasParameterBlock
# Dispatches Imports
from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
# from idaes.power_generation.carbon_capture.piperazine_surrogates.\
#     co2_capture_system import CO2Capture
from dispatches.models.fossil_case.ultra_supercritical_plant.\
    co2_capture_system import CO2Capture


def add_fireside(m):

    m.fs.prop_fluegas = FlueGasParameterBlock()

    boiler_input_dict = {1: '614623556',  # replaced later by boiler
                         'pl': '126983008',  # replaced later by reheater[1]
                         'roof': '96959378',  # replaced later by reheater[2]
                         'flyash': '0.0001',  # flyash mass fraction
                         'NOx': '140'}  # NOx PPM

    m.fs.boiler_fireside = BoilerFireside(
            default={"dynamic": False,
                     "property_package": m.fs.prop_fluegas,
                     "calculate_PA_SA_flows": True,
                     "number_of_zones": 1,
                     "has_platen_superheater": True,
                     "has_roof_superheater": True,
                     "surrogate_dictionary": boiler_input_dict})
    m.fs.boiler_fireside.eq_surr_waterwall_heat.deactivate()
    m.fs.boiler_fireside.eq_surr_platen_heat.deactivate()
    m.fs.boiler_fireside.eq_surr_roof_heat.deactivate()

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                                     m.fs.boiler_fireside.zones,
                                     doc="Waterwall heat duty")
    def eq_main_boiler_heat(b, t, z):
        return (
            b.waterwall_heat[t, z] *
            b.fcorrection_heat_ww[t] ==
            m.fs.boiler.heat_duty[0])

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                                     doc="Reheater heat duty")
    def eq_reheater_1_heat(b, t):
        return (
            b.platen_heat[t] *
            b.fcorrection_heat_platen[t] ==
            m.fs.reheater[1].heat_duty[0])

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                                     doc="Additional reheater heat duty")
    def eq_reheater_2_heat(b, t):
        return (
            b.roof_heat[t] *
            b.fcorrection_heat_ww[t] ==
            m.fs.reheater[2].heat_duty[0])

    # Fixing Coal composition
    m.fs.boiler_fireside.mf_C_coal_dry.fix(0.718471768285811)
    m.fs.boiler_fireside.mf_H_coal_dry.fix(0.0507156542319396)
    m.fs.boiler_fireside.mf_O_coal_dry.fix(0.0791164206018258)
    m.fs.boiler_fireside.mf_N_coal_dry.fix(0.0140876817310943)
    m.fs.boiler_fireside.mf_S_coal_dry.fix(0.0282880649160374)
    m.fs.boiler_fireside.mf_Ash_coal_dry.fix(0.109320410233292)
    m.fs.boiler_fireside.hhv_coal_dry.fix(2.581e+007)
    m.fs.boiler_fireside.frac_moisture_vaporized[:].fix(0.6)
    m.fs.boiler_fireside.mf_H2O_coal_raw[:].fix(0.111367051)  # moisture
    m.fs.boiler_fireside.flowrate_coal_raw[:].fix(100.0)  # kg/s

    m.fs.boiler_fireside.wall_temperature_waterwall[:, :].fix(690)
    m.fs.boiler_fireside.wall_temperature_platen[:].fix(750)
    m.fs.boiler_fireside.wall_temperature_roof[:].fix(650)
    m.fs.boiler_fireside.fcorrection_heat_ww.fix(0.95)
    m.fs.boiler_fireside.fcorrection_heat_platen.fix(0.95)

    # -----------------------------------------------------------
    # Estimates for unit model initialization only
    # -----------------------------------------------------------
    # Approximated flue gas = 21290.6999  # mol/s
    flow_mol_pa = 21290.6999*0.34  # approx. 1/3 as Primary air
    flow_mol_sa = 21290.6999*0.66  # approx. 2/3 as Secondary air

    m.fs.state_args_PA = {
        "flow_mol_comp": {
            "H2O": 0.0078267*flow_mol_pa,
            "CO2": 0.000337339*flow_mol_pa,
            "N2": 0.783994*flow_mol_pa,
            "O2": 0.20784*flow_mol_pa,
            "SO2": 1e-5*flow_mol_pa,
            "NO": 1e-5*flow_mol_pa
        },
        "temperature": 333.15,
        "pressure": 101325.00
    }

    m.fs.state_args_SA = {
        "flow_mol_comp": {
            "H2O": 0.0078267*flow_mol_sa,
            "CO2": 0.000337339*flow_mol_sa,
            "N2": 0.783994*flow_mol_sa,
            "O2": 0.20784*flow_mol_sa,
            "SO2": 1e-5*flow_mol_sa,
            "NO": 1e-5*flow_mol_sa
        },
        "temperature": 650.15,
        "pressure": 101325.00
    }
    # -----------------------------------------------------------

    m.fs.boiler_fireside.primary_air_inlet.pressure[:].fix(101325.00)
    m.fs.boiler_fireside.secondary_air_inlet.pressure[:].fix(101325.00)
    m.fs.boiler_fireside.primary_air_inlet.temperature[:].fix(333.15)
    m.fs.boiler_fireside.secondary_air_inlet.temperature[:].fix(650.15)
    m.fs.boiler_fireside.temperature_coal[:].fix(335.15)
    m.fs.boiler_fireside.flue_gas_outlet.temperature.setub(5000)
    m.fs.boiler_fireside.SR.fix(1.2)
    m.fs.boiler_fireside.ratio_PA2coal.fix(2.45)
    m.fs.boiler_fireside.SR_lf.fix(1.0)
    m.fs.boiler_fireside.deltaP.fix(1000)
    return m


def add_co2capture(m):
    m.fs.co2_capture_unit = CO2Capture()

    # Adding constraints to connect flue_gas_outlet to co2_capture_unit_inlet
    # An arc cannot be used because of the fixed components with capture unit
    # The following are equality constraints for component flows, pressure,
    # and temperature.
    m.fg_comp_list = ['CO2', 'H2O', 'N2', 'O2', 'NO', 'SO2']
    m.css_comp_list = ['CO2', 'H2O', 'N2', 'O2']

    @m.fs.co2_capture_unit.Constraint(m.fs.time,
                                      m.css_comp_list,
                                      doc="Flow equality constraints")
    def eq_flow_mol_comp(b, t, c):
        return (
            m.fs.boiler_fireside.flue_gas_outlet.flow_mol_comp[t, c] ==
            m.fs.co2_capture_unit.inlet.flow_mol_comp[t, c])

    @m.fs.co2_capture_unit.Constraint(m.fs.time,
                                      doc="Fixing Ar flow")
    def eq_ar_flow_mol_comp(b, t):
        return (
            m.fs.co2_capture_unit.inlet.flow_mol_comp[t, 'Ar'] == 0.0089 *
            sum(m.fs.boiler_fireside.flue_gas_outlet.flow_mol_comp[t, c]
                for c in m.fg_comp_list))

    # The Flue gas is assumed to be stripped out of NOx and SOx before capture
    # The inlet to the CO2 capture unit the flue gas temperature is at 303.15 K
    m.fs.co2_capture_unit.inlet.temperature.fix(303.15)  # K

    @m.fs.co2_capture_unit.Constraint(m.fs.time,
                                      doc="Pressure equality constraints")
    def eq_pressure(b, t):
        return (
            m.fs.boiler_fireside.flue_gas_outlet.pressure[t] ==
            m.fs.co2_capture_unit.inlet.pressure[t])

    m.fs.co2_capture_unit.CO2_capture_rate.fix(0.9)  # 90 % CO2 Capture
    m.fs.co2_capture_unit.Pz_mol.fix(5)
    m.fs.co2_capture_unit.lean_loading.fix(0.25)

    # Add a dummy heater block to extract steam for specific reformer duty
    m.fs.ccs_reformer = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    # Add a splitter to take the main steam from boiler for CO2 Capture
    m.fs.ccs_splitter = HelmSplitter(
        default={
            "property_package": m.fs.prop_water
        }
    )
    # Add a mixer to add the ccs exhaust to condenser
    m.fs.ccs_mix = HelmMixer(
        default={
            "momentum_mixing_type": MomentumMixingType.minimize,
            "inlet_list": ["bfpt", "ccs"],
            "property_package": m.fs.prop_water,
        }
    )

    # Add constraint to equate the co2 capture reformer duty with ccs heater
    @m.fs.Constraint(m.fs.time,
                      doc="Reformer duty equality constraint")
    def eq_reformerduty(b, t):
        return (
            0 == m.fs.ccs_reformer.heat_duty[t] +
            m.fs.co2_capture_unit.reboiler_duty[t]*1e6)

    @m.fs.Constraint(m.fs.time)
    def constraint_reformer_out_pressure(b, t):
        return (
            b.ccs_reformer.control_volume.properties_out[t].pressure ==
            b.condenser_mix.main_state[t].pressure
        )

    @m.fs.Constraint(m.fs.time)
    def constraint_reformer_out_enthalpy(b, t):
        return (
            b.ccs_reformer.control_volume.properties_out[t].enth_mol ==
            b.ccs_reformer.control_volume.properties_out[t].
            enth_mol_sat_phase["Liq"]
        )

    # Deactivate the connection from boiler to Turbine 1 and add splitter
    # Also, deactvate the bfpt outlet to condesner to add a mixer
    # for ccs sexhaust steam#
    for arc_s in [m.fs.boiler_to_turb1, m.fs.bfpt_to_condmix]:
        arc_s.expanded_block.enth_mol_equality.deactivate()
        arc_s.expanded_block.flow_mol_equality.deactivate()
        arc_s.expanded_block.pressure_equality.deactivate()

    m.fs.boiler_to_ccsplitter = Arc(
        source=m.fs.boiler.outlet,
        destination=m.fs.ccs_splitter.inlet,
        doc="Connection from boiler to ccs splitter"
    )
    m.fs.ccsplitter_to_turb1 = Arc(
        source=m.fs.ccs_splitter.outlet_1,
        destination=m.fs.turbine[1].inlet,
        doc="Connection from boiler to ccs splitter"
    )
    m.fs.ccsplitter_to_capture = Arc(
        source=m.fs.ccs_splitter.outlet_2,
        destination=m.fs.ccs_reformer.inlet,
        doc="Connection from boiler to ccs splitter"
    )
    # add ccs exhaust steam to condenser
    m.fs.bfpt_to_ccsmix = Arc(
        source=m.fs.bfpt.outlet,
        destination=m.fs.ccs_mix.bfpt
    )
    m.fs.capture_to_ccsmix = Arc(
        source=m.fs.ccs_reformer.outlet,
        destination=m.fs.ccs_mix.ccs
    )
    m.fs.ccsmix_to_condmix = Arc(
        source=m.fs.ccs_mix.outlet,
        destination=m.fs.condenser_mix.bfpt
    )
    TransformationFactory("network.expand_arcs").apply_to(m.fs)

    return m


def initialize_usc_w_capture(m, fileinput=None, outlvl=idaeslog.NOTSET,
                             solver=None, optarg={}):

    iscale.calculate_scaling_factors(m)

    m.fs.boiler.heat_duty[0].fix()
    m.fs.reheater[1].heat_duty[0].fix()
    m.fs.reheater[2].heat_duty[0].fix()

    m.fs.boiler_fireside.initialize(
        state_args_PA=m.fs.state_args_PA,
        state_args_SA=m.fs.state_args_SA)

    m.fs.boiler.heat_duty[0].unfix()
    m.fs.reheater[1].heat_duty[0].unfix()
    m.fs.reheater[2].heat_duty[0].unfix()

    # The initialize method in CO2Capture unit model fixes inlet state
    # and does not unfix it. So, to use the initialize method, the
    # constraints are deactivated before initializing and activated later.
    # The inlet state is unfixed before activating the constraints.
    # TODO: update this section when the initialize method in CO2Capture model
    # is updated.
    m.fs.co2_capture_unit.eq_flow_mol_comp.deactivate()
    m.fs.co2_capture_unit.eq_ar_flow_mol_comp.deactivate()
    m.fs.co2_capture_unit.eq_pressure.deactivate()
    m.fs.co2_capture_unit.inlet.pressure[:].fix(101325)  # Pa (1 atm)

    m.fs.co2_capture_unit.inlet.flow_mol_comp[0, 'CO2'].fix(5328)
    m.fs.co2_capture_unit.inlet.flow_mol_comp[0, 'H2O'].fix(3138)
    m.fs.co2_capture_unit.inlet.flow_mol_comp[0, 'Ar'].fix(341)
    m.fs.co2_capture_unit.inlet.flow_mol_comp[0, 'O2'].fix(1256)
    m.fs.co2_capture_unit.inlet.flow_mol_comp[0, 'N2'].fix(28524)

    m.fs.co2_capture_unit.initialize(outlvl=idaeslog.INFO)

    m.fs.co2_capture_unit.inlet.pressure[:].unfix()  # Pa (1 atm)
    m.fs.co2_capture_unit.inlet.flow_mol_comp[:, 'CO2'].unfix()
    m.fs.co2_capture_unit.inlet.flow_mol_comp[:, 'O2'].unfix()
    m.fs.co2_capture_unit.inlet.flow_mol_comp[:, 'Ar'].unfix()
    m.fs.co2_capture_unit.inlet.flow_mol_comp[:, 'H2O'].unfix()
    m.fs.co2_capture_unit.inlet.flow_mol_comp[:, 'N2'].unfix()
    m.fs.co2_capture_unit.eq_flow_mol_comp.activate()
    m.fs.co2_capture_unit.eq_ar_flow_mol_comp.activate()
    m.fs.co2_capture_unit.eq_pressure.activate()

    propagate_state(m.fs.boiler_to_ccsplitter)
    m.fs.ccs_splitter.split_fraction[:, "outlet_2"].fix(0.24)
    m.fs.ccs_splitter.initialize()
    m.fs.ccs_splitter.split_fraction[:, "outlet_2"].unfix()

    propagate_state(m.fs.ccsplitter_to_capture)
    m.fs.constraint_reformer_out_pressure.deactivate()
    m.fs.eq_reformerduty.deactivate()
    m.fs.constraint_reformer_out_enthalpy.deactivate()
    m.fs.ccs_reformer.heat_duty[0].fix(-853190892.2399481)
    m.fs.ccs_reformer.outlet.pressure[:].fix(6000)
    m.fs.ccs_reformer.inlet.flow_mol.fix(14541.6)
    m.fs.ccs_reformer.initialize()

    m.fs.eq_reformerduty.activate()
    m.fs.constraint_reformer_out_pressure.activate()
    m.fs.constraint_reformer_out_enthalpy.activate()
    m.fs.ccs_reformer.outlet.pressure[:].unfix()
    m.fs.ccs_reformer.heat_duty[0].unfix()
    m.fs.ccs_reformer.inlet.flow_mol.unfix()

    propagate_state(m.fs.bfpt_to_ccsmix)
    propagate_state(m.fs.capture_to_ccsmix)
    m.fs.ccs_mix.initialize()

    propagate_state(m.fs.ccsmix_to_condmix)
    m.fs.condenser_mix.initialize()

    # Increasing the flow through boiler to account for parasitic power
    # required for CO2 capture
    m.fs.boiler.inlet.flow_mol.fix(m.main_flow*2.015)
    
    res = solver.solve(m)

    print("Model Initialization = ",
          res.solver.termination_condition)
    print("**************  USC model w Capture Initialized   ***************")


def build_usc_w_ccs(solver):

    m = usc.build_plant_model()
    usc.initialize(m)
    m = add_fireside(m)
    m = add_co2capture(m)
    assert degrees_of_freedom(m) == 0

    initialize_usc_w_capture(m, solver=solver)
    assert degrees_of_freedom(m) == 0

    return m


if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver("ipopt", optarg)

    m = build_usc_w_ccs(solver)

    results = solver.solve(m, tee=True)
    print('Plant Power (MW) =', value(m.fs.plant_power_out[0]))
