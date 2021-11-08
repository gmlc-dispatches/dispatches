Solar Salt Properties
=====================

This property package supports property calculations for Solar salt in a sensible-heat thermal energy storage system.
Solar salt is a binary mixture for nitrate salts, i.e., 60% by wt. of NaNO3 and 40% by wt. of KNO3.
This property package supports calculations for the following properties for Solar salt within the temperature range of 513 - 853 K [1] [2]:

1. Specific Enthalpy
2. Density
3. Specific Heat Capacity
4. Dynamic Viscosity
5. Thermal Conductivity

**Flow basis**: Mass

**Units**: SI units

**State Variables**: 
The state block has the following 3 state variables:

1. Mass flow rate (kg/s) - `flow_mass`
2. Pressure (Pa) - `pressure`
3. Temperature (K) - `temperature`


References
----------
[1] 2008, Ferri et al., Molten salt mixture properties in RELAP5 code for thermodynamic solar applications. Int. J Ther. Sci., (47) 1676 - 1687


[2] 2015, Chang et al, The design and numerical study of a 2MWh molten salt thermocline tank. Energy Procedia 69, 779 - 789

.. module:: dispatches.models.fossil_case.properties.solarsalt_properties

.. autoclass:: SolarsaltStateBlock
  :members: