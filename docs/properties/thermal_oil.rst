.. _Thermal Oil Property Package:

Thermal Oil
===========

This property package supports property calculations for Thermal oil in a sensible-heat thermal energy storage system.
This property package supports calculations for the following properties for Thermal Oil within the temperature range of 0 - 345 C [1]:

1. Specific Enthalpy
2. Density
3. Specific Heat Capacity
4. Kinematic Viscosity
5. Thermal Conductivity
6. Dynamic Viscosity

**Flow basis**: Mass

**Units**: SI units

**State Variables**: 
The state block has the following 3 state variables:

1. Mass flow rate (kg/s) - `flow_mass`
2. Pressure (Pa) - `pressure`
3. Temperature (K) - `temperature`

References
----------
[1] Therminol 66, High Performance Highly Stable Heat Transfer Fluid (0C to 345C), Solutia.


.. module:: dispatches.models.fossil_case.properties.thermaloil_properties

.. autoclass:: ThermalOilStateBlock
  :members: