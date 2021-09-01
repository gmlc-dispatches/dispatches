Hydrogen Tank
================

The DISPATCHES Hydrogen Tank Model represents a steady state implementation of a compressed hydrogen gas tank. This tank model supports tank filling and emptying operations for a fixed duration assuming a constant flow.

Degrees of Freedom
------------------

The Hydrogen Tank Model has 9 degrees of freedom. The degrees of freedom include the inlet state variables (flow, pressure, and temperature), previous state variables (temperature and pressure), tank dimensions (tank diameter and tank length), time duration, and outlet flow (for tank filling scenario flow = 0). By default, the tank is set to be adiabatic by fixing heat duty = 0.


Model Structure
---------------

The Hydrogen Tank Model consists of a single `ControlVolume0D` (named `control_volume`) with 2 states (`properties_in` and `properties_out`) and 2 ports (named inlet and outlet). In addition, the tank model has another state named `previous_state` that denotes the state of the tank at the beginning of time period given by `dt`. The tank model then computes holdup terms integrated over the time `dt`. Custom material and energy balance are written to account for integrated holdup terms. Finally, an internal energy balance computes the outlet temperature.


Governing Equations
-------------------

Internal enery balance for tank temperature calculation at the end of time step :math:`dt`
.. math:: (`T_out_t` - `T_ref`) * `C_v_out` * `M_prev_tpj` + `F_in_tpj` * `dt_t` = (`T_prev_t` - `T_ref`) * `C_v_prev` * `M_prev_tpj` + (`T_in_t` - `T_ref`) * `C_p_in` * `F_in_tpj` * `dt_t`

where,
:math:`T_in_t`, :math:`T_out_t`, :math:`T_prev_t`, :math:`T_ref` are Temperature at inlet, outlet, previous state, and reference conditions respectively
:math:`C_v_out`, :math:`C_v_prev` are constant volume specific heat capacity at outlet and previous state
:math:`C_p_in` is constant pressure specific heat capacity at inlet
:math:`F_in_tpj` is inlet flow

Variables Used
--------------

The Hydrogen Tank Model uses the follow variables:

=================== ========================== ============================================================================
Variable            Name                       Notes
=================== ========================== ============================================================================
:math:`V_t`         volume                     tank volume
:math:`Q_t`         heat_duty                  heat duty (default = 0,i.e., adiabatic)
:math:`D`           tank_diameter              diameter of tank
:math:`L`           tank_length                length of tank
:math:`dt`          dt                         time step
:math:`dM_tpj`      material_accumulation      average material accumulation term over :math:`dt`
:math:`dE_tp`       energy_accumulation        average energy accumulation term over :math:`dt`
:math:`M_tpj`       material_holdup            material holdup
:math:`E_tp`        energy_holdup              energy holdup
:math:`M_prev_tpj`  previous_material_holdup   previous state material holdup
:math:`E_prev_tp`   previous_energy_holdup     previous state energy holdup
=================== ========================== ============================================================================

.. module:: dispatches.models.nuclear_case.unit_model.hydrogen_tank

