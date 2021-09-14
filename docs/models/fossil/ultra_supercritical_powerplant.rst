Ultra-Supercritical Power Plant
===============================

The DISPATCHES Ultra-Supercritical Power Plant Model is an example flowsheet for a pulverized coal-fired ultra-supercritical power plant. This model simulates a plant producing ~436 MW of gross power.

.. image:: ../../images/ultra_supercritical_powerplant.png

Abbreviations
-------------

================== ================================
Acronym            Name
================== ================================
RH                 Reheater (RH1 & RH2)
BFPT               Boiler Feed Water Pump Turbine
FWH                Feed Water Heaters (FWH1 - FWH9)
BFW                Boiler Feed Water
USC                Ultra Supercritical
:math:`F_{cond}`   Condenser Flow Out (mol/s)
:math:`F_{boiler}` Boiler Flow In (mol/s)
================== ================================

Model Structure
---------------

The Ultra Supercritical Power Plant Model consists the following  models from the idaes/power_generation unit model library in addition to the IAPWS property package for steam and water.

=========================== ============================================================
Unit Model                  Units in the flowsheet
=========================== ============================================================
`HelmTurbineStage`          Turbines (T1 to T11) and BFPT
`HelmSplitter`              Turbine Splitters
`Heater`                    Boiler components, i.e., Boiler, Reheater_1, and Reheater_2
`HelmMixer`                 Mixers (including Condensate Mixer and Deaerator)
`HelmIsentropicCompresssor` Pumps, i.e., Condenser Pump, Booster Pump, and BFW Pump
`HeatExchanger`             Condenser and Feedwater Heaters, FWH1 - FWH9
=========================== ============================================================

Degrees of Freedom
------------------

The Ultra Supercritical Power Plant Model has 2 degrees of freedom, i.e., feedwater flow (`boiler.inlet.flow_mol`) and feedwater pressure (`boiler.outlet.pressure`)


Notable Variables
-----------------

=============== ========================================================
Variable Name   Description
=============== ========================================================
`PlantPowerOut` Net power out from the plant in MW
`PlantHeatDuty` Total boiler heat duty (i.e., Boier, RH1 & RH2) in MW_th
=============== ========================================================


Notable Constraints
-------------------

1) Boiler temperature out is set to be 866 K, i.e.

.. math:: Unit.Temperature_{out, t} = 866
where, `Unit` is in [`Boiler`, `Reheater_1`, `Reheater_2`]

2) Plant_Power_Out is given by the total turbine mechanical work, i.e.,

.. math:: PlantPowerOut = \sum_{Unit}{Unit.MechanicalWork_{t}}
where, `Unit` is in {T1 : T11]

3) Plant_Heat_Duty is given as the sum of heat duties for Boiler units,

.. math:: PlantHeatDuty = \sum_{Unit}{Unit.HeatDuty_{t}}

where, `Unit` is in [`Boiler`, `Reheater_1`, `Reheater_2`]




.. automodule:: dispatches.models.fossil_case.ultra_supercritical_plant.ultra_supercritical_powerplant



