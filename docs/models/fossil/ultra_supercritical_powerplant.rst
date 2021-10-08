Ultra-Supercritical Power Plant
===============================

The DISPATCHES Ultra-Supercritical Power Plant Model is an example flowsheet for a pulverized coal-fired ultra-supercritical power plant. This model simulates a plant producing ~436 MW of gross power.

.. image:: ../../images/ultra_supercritical_powerplant.png

Abbreviations
-------------

================== ================================
Acronym            Name
================== ================================
:math:`RH`         Reheater (:math:`RH1` and :math:`RH2`)
:math:`BFPT`       Boiler Feed Water Pump Turbine
:math:`FWH`        Feed Water Heaters (:math:`FWH1` to :math:`FWH9`)
:math:`BFW`        Boiler Feed Water
:math:`USC`        Ultra-Supercritical
:math:`F_{cond}`   Condenser Flow Out (mol/s)
:math:`F_{boiler}` Boiler Flow In (mol/s)
================== ================================

Model Structure
---------------

The ultra-supercritical Power Plant Model consists the following  models from the idaes/power_generation unit model library in addition to the IAPWS property package for steam and water.

================================= ============================================================
Unit Model                        Units in the flowsheet
================================= ============================================================
:math:`HelmTurbineStage`          Turbines (:math:`T1` to :math:`T11`) and :math:`BFPT`
:math:`HelmSplitter`              Turbine Splitters
:math:`Heater`                    Boiler components, i.e., :math:`Boiler`, :math:`RH1`, and :math:`RH2`
:math:`HelmMixer`                 Mixers (including :math:`Condensate Mixer` and :math:`Deaerator`)
:math:`HelmIsentropicCompresssor` Pumps, i.e., :math:`Condenser Pump`, :math:`Booster Pump`, and :math:`BFW Pump`
:math:`HeatExchanger`             Condenser and Feedwater Heaters, :math:`FWH1` to :math:`FWH9`
================================= ============================================================

Degrees of Freedom
------------------

The ultra-supercritical Power Plant Model has 2 degrees of freedom, i.e., feedwater flow (:math:`boiler.inlet.flow_-mol`) and feedwater pressure (:math:`boiler.outlet.pressure`)


Notable Variables
-----------------

===================== ========================================================
Variable Name         Description
===================== ========================================================
:math:`PlantPowerOut` Net power out from the plant in MW
:math:`PlantHeatDuty` Total boiler heat duty (i.e., :math:`Boiler`, :math:`RH1`, and :math:`RH2`) in MWth
===================== ========================================================


Notable Constraints
-------------------

1) Boiler temperature out is set to be 866 K, i.e.

.. math:: Unit.Temperature_{out, t} = 866

where, :math:`Unit` is in :math:`[Boiler, RH1, RH2]`

2) :math:`PlantPowerOut` is given by the total turbine mechanical work, i.e.,

.. math:: PlantPowerOut = \sum_{Unit}{Unit.MechanicalWork_{t}}

where, :math:`Unit` is in :math:`[T1 : T11]`

3) :math:`PlantHeatDuty` is given as the sum of heat duties for Boiler units,

.. math:: PlantHeatDuty = \sum_{Unit}{Unit.HeatDuty_{t}}

where, :math:`Unit` is in :math:`[Boiler, RH1, RH2]`




.. automodule:: dispatches.models.fossil_case.ultra_supercritical_plant.ultra_supercritical_powerplant



