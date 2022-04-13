Ultra-Supercritical Power Plant
===============================

The DISPATCHES Ultra-Supercritical Power Plant model is an example flowsheet for a pulverized coal-fired ultra-supercritical power plant. This model simulates a plant producing 436 MW of gross power.

.. image:: ../../images/ultra_supercritical_powerplant.png
    :align: center
	   
Abbreviations
-------------

================== ================================
Acronym            Name
================== ================================
:math:`RH`         Reheater (:math:`RH_1` and :math:`RH_2`)
:math:`T`          Turbine (:math:`T_1` to :math:`T_{11}`)
:math:`BFPT`       Boiler Feed Water Pump Turbine
:math:`FWH`        Feed Water Heaters (:math:`FWH_1` to :math:`FWH_9`)
:math:`CM`         Condensate Mixer
:math:`CP`         Condenser Pump
:math:`BP`         Booster Pump
:math:`BFWP`       Boiler Feed Water Pump
:math:`DA`         Deaerator
:math:`BFW`        Boiler Feed Water
:math:`F_{cond}`   Condenser Flow Outlet (mol/s)
:math:`F_{boiler}` Boiler Flow Inlet (mol/s)
================== ================================

Model Structure
---------------

The ultra-supercritical Power Plant model consists of the following models from the idaes/power_generation unit model library in addition to the IAPWS property package for steam and water.

================================= ============================================================
Unit Model                        Units in the flowsheet
================================= ============================================================
:math:`HelmTurbineStage`          Turbines (:math:`T_1` to :math:`T_{11}`) and :math:`BFPT`
:math:`HelmSplitter`              Turbine Splitters
:math:`Heater`                    Boiler components, i.e., :math:`Boiler`, :math:`RH_1`, and :math:`RH_2`
:math:`HelmMixer`                 Mixers (including :math:`CM` and :math:`DA`)
:math:`HelmIsentropicCompresssor` Pumps (including :math:`CP`, :math:`BP`, and :math:`BFWP`)
:math:`HeatExchanger`             Condenser and Feed Water Heaters (:math:`FWH_1` to :math:`FWH_9`)
================================= ============================================================

Degrees of Freedom
------------------

The ultra-supercritical Power Plant model has 2 degrees of freedom:

1) Boiler feed water flow (:math:`boiler.inlet.flow_-mol`)

2) Boiler feed water pressure (:math:`boiler.outlet.pressure`)



Notable Variables
-----------------

===================== ========================================================
Variable Name         Description
===================== ========================================================
:math:`PlantPowerOut` Net power out from the plant in MW
:math:`PlantHeatDuty` Total boiler heat duty (i.e., :math:`Boiler`, :math:`RH_1`, and :math:`RH_2`) in MWth
===================== ========================================================


Notable Constraints
-------------------

1) The outlet temperature of the boiler components is set to be 866 K, as shown in the following equation, where :math:`Unit` represents :math:`Boiler, RH_1`, and :math:`RH_2`:

.. math:: Unit.outlet.temperature_t = 866

2) :math:`PlantPowerOut` is given by the total turbine mechanical work, as shown in the following equation:

.. math:: PlantPowerOut_t = \sum^{11}_{i=1}{T_i.mechanical_-work_t}

3) :math:`PlantHeatDuty` is given as the summation of the heat duties of boiler components as shown in the following equation, where :math:`Unit` is in :math:`[Boiler, RH_1, RH_2]`:

.. math:: PlantHeatDuty_t = \sum_{Unit}{Unit.heat_-duty_t}




.. automodule:: dispatches.models.fossil_case.ultra_supercritical_plant.ultra_supercritical_powerplant



