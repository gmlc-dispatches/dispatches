Multiperiod Integrated Ultra-Supercritical Power Plant
======================================================

The Multiperiod Integrated Ultra-Supercritical Power Plant is an example model to determine the optimal schedule and operation of a pulverized coal-fired ultra-supercritical power plant integrated with thermal energy storage system for given electricity prices over a time horizon. The multiperiod model comprises multiple instances of the Integrated Ultra-Supercritical Power Plant model, which integrates a charge and discharge heat exchanger to the power plant. A detailed description and flowsheet for the integrated ultra-supercritical model are given in Integrated Ultra-Supercritical Power Plant.


Model Structure
---------------

The multiperiod power plant model is developed by creating multiple instances of the steady-state nonlinear power plant model described in Integrated Ultra-Supercritical Power Plant, with every instance indexed for a time period, along with two coupling variables to link each time step. The coupling variables are: the optimal power produced by the power plant and the amount of storage material available at the end of each time period. A scheme showing how the multiperiod model is constructed is shown in the figure below:

.. image:: ../../images/multiperiod_integrated_ultra_supercritical_powerplant.png
	   :align: center



Degrees of Freedom
------------------

The multiperiod model has a total of 4 :math:`n` degrees of freedom, where :math:`n` represents the number of time periods. The degrees of freedom for each time period are listed below:

1) Boiler feed water flow (:math:`boiler.inlet.flow_-mol`),
 
2) Charge splitter flow to storage (:math:`charge_-splitter.outlet_-2.flow_-mol`)
   
3) Discharge splitter flow to storage (:math:`discharge_-splitter.outlet_-2.flow_-mol`),

4) Cooler enthalpy in charge system (:math:`cooler.outlet.enth_-mol`)



Notable Variables
-----------------

========================= ========================================================
Variable Name             Description
========================= ========================================================
:math:`NetPower_t`        Net power out from the power plant at each time period  in MW
:math:`SaltInventory_t`   Solar salt inventory at each time period in kg
========================= ========================================================


Notable Constraints
-------------------

1) Ramping in the power plant is limited to a given ramping value :math:`ramp_-rate` by including the following equations:

.. math:: NetPower_{t-1}  - ramp_-rate \leq NetPower_t
.. math:: NetPower_{t-1}  + ramp_-rate \geq NetPower_t

2) The salt inventory :math:`SaltInventory` is given by the hot salt and total salt material balances, the latter involving a fixed amount of salt :math:`total_-salt`. The material balances are shown in the following equations:

.. math:: HotSaltInventory_t = HotSaltInventory_{t-1} + F^{charge}_{salt, outlet} - F^{discharge}_{salt, inlet}
.. math:: total_-salt = HotSaltInventory_t + ColdSaltInventory_t

Note that the notable constraints and variables in the multiperiod model also consider the notable variables and constraints given in Integrated Ultra-Supercritical Power Plant model.

.. automodule:: dispatches.models.fossil_case.ultra_supercritical_plant.storage.usc_storage_nlp_mp




