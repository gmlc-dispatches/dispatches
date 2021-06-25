Hydrogen Turbine
================

The DISPATCHES Hydrogen Turbine Model


.. image:: ../../images/H2_turbine.png


Degrees of Freedom
------------------

The Hydrogen Turbine Model has 13 degrees of freedom.


Model Structure
---------------

The Hydrogen Turbine Model consists of a Compressor --> Stoichiometric Reactor --> Turbine. 
The Hydrogen is assumed to be compressed alongside the air that is flowing into the compressor 
rather than having a separate fuel injection system. 


Variables Used
--------------

The Hydrogen Turbine Model uses the follow variables:

=================== ============================================== ============================================================================
Variable            Name                                           Notes
=================== ============================================== ============================================================================
:math:`m_flow`      compressor.inlet.flow_mol[0]                   Total Molar flow rate into the inlet of the Compressor
:math:`T_inlet`     compressor.inlet.temperature[0]                Inlet Temperature of stream into the Compressor of the hydrogen turbine
:math:`P_inlet`     compressor.inlet.pressure[0]                   Inlet Pressure into the Compressor
:math:`w_Oxygen`    compressor.inlet.mole_frac_comp[0, "oxygen"]   Mole fraction of oxygen entering the Compressor Inlet
:math:`w_Argon`     compressor.inlet.mole_frac_comp[0, "argon"]    Mole fraction of argon entering the Compressor Inlet
:math:`w_nitrogen`  compressor.inlet.mole_frac_comp[0, "nitrogen"] Mole fraction of nitrogen entering the Compressor Inlet
:math:`w_water`     compressor.inlet.mole_frac_comp[0, "water"]    Mole fraction of water entering the Compressor Inlet
:math:`w_hydrogen`  compressor.inlet.mole_frac_comp[0, "hydrogen"] Mole fraction of hydrogen entering the Compressor Inlet
:math:`dP_comp`     compressor.deltaP                              Pressure change across the compressor
:math:`\eta_comp`   compressor.efficiency_isentropic               Compressor isentropic efficiency Value
:math:`\epsilon_RX` stoic_reactor.conversion                       Conversion Rate inside the stoichiometric reactor. Value between [0,1]
:math:`dP_turb`     turbine.deltaP                                 Pressure change across the turbine
:math:`\eta_turb`   turbine.efficiency_isentropic                  Turbine isentropic efficiency Value
=================== ============================================== ============================================================================

.. module:: dispatches.models.nuclear_case.unit_models.hydrogen_turbine


Hydrogen Turbine Costs
----------------------
Hydrogen Turbines are a cutting edge technology that looks to modify the years of work on natural gas turbines to burn hydrogen as opposed to natural gas. 
As such, baseline costs of natural gas turbines with a modifier placed on top of them is considered. 

========================= ===============================================================================================
Natural Gas Capital Costs Value
========================= ===============================================================================================
Capital Costs:            $947/kW - $1061/kW
Fixed OPEX:               5.2 - 10.1 Million/Year
Variable OPEX:            $4.25/MWh - $4.29/MWh
Scaling Factor:           0.72 - 0.78 [0.72 if the entire plant is scaled, 0.78 if only the turbine is scaled]
========================= ===============================================================================================