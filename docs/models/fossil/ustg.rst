Ultra-Supercritical Thermal Generator
=====================================

The DISPATCHES Ultra Supercritical Thermal Generator Model is an example flowsheet for a pulverized coal-fired ultra-supercritical power plant. This model simulates a plant producing ~436 MW of gross power.

Degrees of Freedom
------------------

The Ultra Supercritical Thermal Generator Model has 2 degrees of freedom, i.e., feedwater flow (`boiler.inlet.flow_mol`) and feedwater pressure (`boiler.outlet.pressure`)


Model Structure
---------------

The Ultra Supercritical Thermal Generator Model consists the following unit models from the power generation library and IAPWS property package for steam and water.

HelmTurbineStage: turbine
HelmSplitter: turbine_splitter
Heater: boiler, reheater, condenser
HelmMixer: condenser_mix, fwh_mixer, deaerator
HelmIsentropicCompresssor: cond_pump, booster, bfp
HeatExchanger: fwh
HelmTurbineStage: bfpt



.. module:: dispatches.models.fossil_case.unit_model.ustg



