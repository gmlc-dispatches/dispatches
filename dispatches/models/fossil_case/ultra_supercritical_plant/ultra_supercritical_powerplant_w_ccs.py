"""
This is an example to study GibbsReactor module.
"""

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc

# IDAES Imports
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.power_generation.unit_models.boiler_fireside import BoilerFireside
# from fluegas_ideal_vap import get_fluegas_properties
from idaes.core.util import get_solver
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# from idaes.generic_models.properties.core.generic.generic_property import (
#     GenericParameterBlock)
import ultra_supercritical_powerplant as usc
from co2_capture_system import CO2Capture

# from idaes.power_generation.carbon_capture.piperazine_surrogates.\
#     co2_capture_system import CO2Capture
from idaes.power_generation.properties import FlueGasParameterBlock

def add_fireside(m):
    # fluegas_config = get_fluegas_properties(
    #         components=['CO2', "H2O", "N2", "NO", "O2", "SO2"])

    # m.fs.prop_fluegas = GenericParameterBlock(default=fluegas_config)
    m.fs.prop_fluegas = FlueGasParameterBlock()

    boiler_input_dict = {1: '614623556', # 'm.fs.boiler.heat_duty[0].value'
                         'pl': '126983008', # 'm.fs.reheater[1].heat_duty[0].value'
                         'roof': '96959378', # 'm.fs.reheater[2].heat_duty[0].value'
                         'flyash': '0.0001',  # flyash or unburned carbon mass fraction
                         'NOx': '140'}  # NOx PPM

    m.fs.boiler_fireside = BoilerFireside(
            default={"dynamic": False,
                     "property_package": m.fs.prop_fluegas,
                     "calculate_PA_SA_flows": True,
                     "number_of_zones": 1,
                     "has_platen_superheater": True,
                     "has_roof_superheater": True, #})
                     "surrogate_dictionary": boiler_input_dict})
    m.fs.boiler_fireside.eq_surr_waterwall_heat.deactivate()
    m.fs.boiler_fireside.eq_surr_platen_heat.deactivate()
    m.fs.boiler_fireside.eq_surr_roof_heat.deactivate()

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                     m.fs.boiler_fireside.zones,
                     doc="Surrogate model for heat loss in boiler")
    def eq_main_boiler_heat(b, t, z):
        return (
            b.waterwall_heat[t, z] *
            b.fcorrection_heat_ww[t] ==
            m.fs.boiler.heat_duty[0])

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                     doc="Surrogate model for heat loss in reheater 1")
    def eq_reheater_1_heat(b, t):
        return (
            b.platen_heat[t] *
            b.fcorrection_heat_platen[t] ==
            m.fs.reheater[1].heat_duty[0])

    @m.fs.boiler_fireside.Constraint(m.fs.time,
                     doc="Surrogate model for heat loss in reheater 2")
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
    m.fs.boiler_fireside.mf_H2O_coal_raw[:].fix(0.111367051)  # moisture content
    m.fs.boiler_fireside.flowrate_coal_raw[:].fix(100.0)  # kg/s

    m.fs.boiler_fireside.wall_temperature_waterwall[:, :].fix(690)
    m.fs.boiler_fireside.wall_temperature_platen[:].fix(750)
    m.fs.boiler_fireside.wall_temperature_roof[:].fix(650)
    m.fs.boiler_fireside.fcorrection_heat_ww.fix(0.95)
    m.fs.boiler_fireside.fcorrection_heat_platen.fix(0.95)

    # SCPC simulation approx flue gas = 21290.6999  # mol/s
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

    # for i in m.fs.prop_fluegas.component_list:
    #     m.fs.boiler.primary_air_inlet.flow_mol_comp[:, i].\
    #         fix(state_args_PA["flow_mol_comp"][i])
    #     m.fs.boiler.secondary_air_inlet.flow_mol_comp[:, i].\
    #         fix(state_args_SA["flow_mol_comp"][i])
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
    m.fs.boiler_to_capture = Arc(
        source=m.fs.boiler_fireside.flue_gas_outlet,
        destination=m.fs.co2_capture_unit.inlet
    )
    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)

    return m

def initialize_usc_w_capture(m, fileinput=None, outlvl=idaeslog.NOTSET,
               solver=None, optarg={}):

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver(solver, optarg)

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

    propagate_state(m.fs.boiler_to_capture)
    m.fs.co2_capture_unit.initialize()
    
    res = solver.solve(m)
    print("Model Initialization = ",
          res.solver.termination_condition)
    print("**************  USC model w Capture Initialized   ***************")

def build_usc_w_ccs():

    m = usc.build_plant_model()
    solver = usc.initialize(m)
    m = add_fireside(m)
    m = add_co2capture(m)

    initialize_usc_w_capture(m)

    return m

if __name__ == "__main__":

    optarg = {
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    solver = get_solver("ipopt", optarg)

    m = build_usc_w_ccs()

    results = solver.solve(m, tee=True)
