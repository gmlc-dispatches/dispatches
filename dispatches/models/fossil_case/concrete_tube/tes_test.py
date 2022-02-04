from pyomo.environ import (ConcreteModel,
                           units as pyunits)
from idaes.core import FlowsheetBlock
from tube_concrete_model import ConcreteTES
from idaes.generic_models.properties import iapws95
from idaes.core.util.model_statistics import degrees_of_freedom

if __name__ == '__main__':
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.props_water = iapws95.Iapws95ParameterBlock(default={
        "phase_presentation": iapws95.PhaseType.LG})

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
        "flow_mol_charge": 0.00958 * 1000 / 18.01528,
        "inlet_pressure_charge": 19600000,
        "inlet_temperature_charge": 853.92699435,
        "flow_mol_discharge": 0 / 18.01528,
        "inlet_pressure_discharge": 8.5e5, #15 MPa
        "inlet_temperature_discharge": 355
        }

    data2 = {
        "delta_time": 1800,
        "time_periods": 2,
        "segments": 20,
        "deltaP": 0,
        "concrete_init_temp": [840.7, 822.8578947, 805.0157895, 787.1736842, 769.331579, 751.4894737, 733.6473684,
                               715.8052632, 697.9631579, 680.1210526, 662.2789474, 644.4368421, 626.5947368,
                               608.7526316, 590.9105263, 573.0684211, 555.2263158, 537.3842105, 519.5421053, 501.7],
        "tube_length":  81.70,
        "tube_diameter":  0.0114,
        "number_tubes": 10000,
        "concrete_area": 0.00601,
        "concrete_conductivity": 1,
        "concrete_density": 2240,
        "concrete_specific_heat": 900,
        # This data is used for initialization, concrete_final_temp is recalculated during the initialization step
        "concrete_final_temp": [852.9795784, 838.7163398, 823.4355578, 807.5334959, 791.222075, 774.658994,
                                757.9645604, 741.2346136, 724.550712, 707.9889509, 691.627446, 675.5540523,
                                659.8784824, 644.7230735, 630.2106417, 616.5177047, 603.8893673, 592.7112727,
                                582.9675115, 574.0510003],
        "flow_mol_charge": 0.00747 * 1000 / 18.01528,
        "inlet_pressure_charge": 19700000,
        "inlet_temperature_charge": 866.0965262,
        "flow_mol_discharge": 0.00747 * 1000 / 18.01528,
        "inlet_pressure_discharge": 19700000,
        "inlet_temperature_discharge": 866.0965262,
        }

    data = data1
    inlet_enthalpy_charge = iapws95.htpx(T=data["inlet_temperature_charge"] * pyunits.K,
                                         P=data["inlet_pressure_charge"] * pyunits.Pa)

    inlet_enthalpy_discharge = iapws95.htpx(T=data["inlet_temperature_discharge"] * pyunits.K,
                                            P=data["inlet_pressure_discharge"] * pyunits.Pa)

    m.fs.tes = ConcreteTES(default={"model_data": data, "property_package": m.fs.props_water})

    for p in m.fs.tes.time_periods:
        m.fs.tes.period[p].tube_charge.tube_inlet.flow_mol[0].fix(data["flow_mol_charge"])
        m.fs.tes.period[p].tube_charge.tube_inlet.pressure[0].fix(data["inlet_pressure_charge"])
        m.fs.tes.period[p].tube_charge.tube_inlet.enth_mol[0].fix(inlet_enthalpy_charge)

        m.fs.tes.period[p].tube_discharge.tube_inlet.flow_mol[0].fix(data["flow_mol_discharge"])
        m.fs.tes.period[p].tube_discharge.tube_inlet.pressure[0].fix(data["inlet_pressure_discharge"])
        m.fs.tes.period[p].tube_discharge.tube_inlet.enth_mol[0].fix(inlet_enthalpy_discharge)

    print(degrees_of_freedom(m))
    m.fs.tes.initialize()

    print('End of the run!')
