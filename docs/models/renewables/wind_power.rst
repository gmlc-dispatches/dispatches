.. _windpower:

Wind Power
============

The DISPATCHES Wind Power Model calculates the electricity production from a wind farm using wind resource data and
the PySAM Windpower module. The model assumes a single turbine with a powercurve from the ATB Turbine 2018 Market Average.
Wake effects are not modeled, and PySAM's default losses are assumed. The power output of the entire farm of any size is
calculated by scaling the output from the single turbine.

Degrees of Freedom
------------------

The Wind Power Model has 0 degrees of freedom.


Model Structure
---------------

The Wind Power Model uses the wind resource data and the single turbine to calculate a `capacity_factor` that is then
used to scale the electricity output of a wind farm of any size. The wind resource data is provided via the ConfigBlock
as a `resource_probability_density`, a probability density function of wind speed [m/s] and wind direction [degrees
clockwise from N] indexed by time. The `setup_atb_turbine` function provides turbine parameters to PySAM and can be
modified to simulate a turbine with a different hub height and power curve.


Variables
--------------

The Wind Power Model uses the following variables:

======================= ===================== ===========================================
Variable Name           Symbol                       Notes
======================= ===================== ===========================================
system_capacity         :math:`S`
capacity_factor         :math:`C_{t}`         Parameter determined via PySAM simulation
electricity             :math:`E_{t}`         Electricity to outlet
======================= ===================== ===========================================

Constraints
---------------
Electricity output scales with system_capacity:

.. math:: E_{t} = S \times C_{t}

Wind Power Class
----------------

.. module:: dispatches.models.renewables_case.wind_power

.. autoclass:: Wind_Power
  :members:
