.. _hydrogentank:

Hydrogen Tank
================

The DISPATCHES Hydrogen Tank Model represents a quasi steady state implementation of a compressed hydrogen gas tank. This tank model supports tank filling and emptying operations for a fixed duration assuming a constant flow.

Degrees of Freedom
------------------

The Hydrogen Tank Model has 5 degrees of freedom and they are previous state variables (temperature and pressure), tank dimensions (tank diameter and tank length), and time duration. In addition, the model needs a defined inlet-state and an operating scenario in terms of the outlet flow (e.g., outlet flow = 0 when tank is filled). By default, the tank is set to be adiabatic by fixing heat duty = 0.


Model Structure
---------------

The Hydrogen Tank Model consists of a single `ControlVolume0D` (named `control_volume`) with 2 states (`properties_in` and `properties_out`) and 2 ports (named inlet and outlet). In addition, the tank model has another state named `previous_state` that denotes the state of the tank at the beginning of time period given by `dt`. The tank model then computes holdup terms integrated over the time `dt`. Custom material and energy balance are written to account for integrated holdup terms. Finally, an internal energy balance computes the outlet temperature.


Governing Equations
-------------------

`previous_state` material holdup rule:

.. math:: M_{prev, t, p , j} = V_{t} \times y_{t, p} \times \rho_{prev, t, p, j}

`previous_state` energy holdup rule:

.. math:: M_{prev, t, p , j} = \sum_j{M_{prev, t, p, j}} \times U_{prev, t, p}

Material balance equation:

.. math:: dM_{t, p , j} = F_{in, t, p, j} - F_{out, t, p, j}

Material holdup calculation:

.. math:: M_{t, p , j} = V_{t} \times y_{t, p} \times \rho_{out, t, p, j}

Material holdup integration over the time step :math:`dt` :

.. math:: M_{t, p , j} = dt_{t} \times dM_{t, p, j} + M_{prev, t, p, j}

Internal enery balance at the end of time step :math:`dt` :

.. math::  E_{t, p} = E_{prev, t, p} + dt_{t} \times (F_{in, t, p, j} \times H_{in, t, p, j} - F_{out, t, p, j} \times H_{out, t, p, j})

Energy holdup calculation:

.. math::  E_{t, p} = \sum_j{M_{t, p, j}} \times U_{t, p}

Energy accumulation:

.. math::  \sum_p{dE_{t, p}} \times dt_{t} = \sum_p{E_{t, p}} - \sum_p{E_{t, p}}

where,
:math:`rho_{t, p}` is the density term
:math:`U_{t, p, j}` is the specific internal energy term
:math:`E_{t, p}` is the energy holdup term
:math:`y_{t, p}` is the phase fraction
:math:`H_{in, t, p, j}` is the specific inlet enthalpy
:math:`H_{out, t, p, j}` is the specific outlet enthalpy
:math:`F_{in, t, p, j}` is the inlet flow
:math:`F_{out, t, p, j}` is the outlet flow

Variables Used
--------------

The Hydrogen Tank Model uses the follow variables:

========================== ========================== ====================================================
Variable                   Name                       Notes
========================== ========================== ====================================================
:math:`V_{t}`              volume                     tank volume
:math:`Q_{t}`              heat_duty                  heat duty (default = 0,i.e., adiabatic)
:math:`D`                  tank_diameter              diameter of tank
:math:`L`                  tank_length                length of tank
:math:`dt_{t}`             dt                         time step
:math:`dM_{t, p, j}`       material_accumulation      average material accumulation term over :math:`dt`
:math:`dE_{t, p}`          energy_accumulation        average energy accumulation term over :math:`dt`
:math:`M_{t, p, j}`        material_holdup            material holdup
:math:`E_{t, p}`           energy_holdup              energy holdup
:math:`M_{prev, t, p, j}`  previous_material_holdup   previous state material holdup
:math:`E_{prev, t, p}`     previous_energy_holdup     previous state energy holdup
========================== ========================== ====================================================

.. module:: dispatches.models.nuclear_case.unit_models.hydrogen_tank

.. autoclass:: HydrogenTank
  :members:
