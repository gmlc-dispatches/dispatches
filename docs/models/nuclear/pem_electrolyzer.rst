PEM Electrolyzer
================

The DISPATCHES PEM Electrolyzer Model represents a simple proton-exchange membrane electrolyzer that operates with a
fixed efficiency in converting electricity to hydrogen gas. The outlet state is determined via an H2 property package.

This file here is a synopsis of the PEM system. The full description with arguments and details on the class are available in 
the renewable case documentation. 

Degrees of Freedom
------------------

The PEM Electrolyzer Model has 0 degrees of freedom.


Model Structure
---------------

The PEM Electrolyzer Model consists of a fixed conversion rate between the inlet electricity and the outlet flow rate of
hydrogen gas.


Variables
--------------

The PEM Electrolyzer Model Model uses the following variables:

========================= ===================== ===========================================
Variable Name             Symbol                Notes
========================= ===================== ===========================================
electricity_to_mol        :math:`\eta`
electricity               :math:`I_{t}`
outlet_state              :math:`O_{t, *}`      state variables such as mol flow, temperature, pressure
========================= ===================== ===========================================

Constraints
---------------
Outlet flow of hydrogen gas depends on efficiency:

.. math:: O_{t, f} = I_{t} \times \eta


