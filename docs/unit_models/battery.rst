.. _battery:

Battery
=======

The DISPATCHES Battery Model represents electricity energy storage with efficiencies for charge and discharge
as well as capacity degradation with cycling. Electricity inflows and outflows determine the state-of-charge and energy
throughput of the battery.

Degrees of Freedom
------------------

The Battery Model has X degrees of freedom.


Model Structure
---------------

The Battery Model uses the inlet and outlet electricity to determine how the stored energy and storage
capacity of the battery changes from the user-defined initial conditions. The initial conditions consist of the
`initial_state_of_charge` and `initial_energy_throughput`. A parameter, `dt`, is required to convert between electricity
flows and stored energy. The `degradation_rate` describes how the storage capacity of the battery decreases with
use, accounted for by `energy_throughput`.


Variables
--------------

The Battery Model uses the following variables:

========================= ===================== ===========================================
Variable Name             Symbol                Notes
========================= ===================== ===========================================
nameplate_power           :math:`P`
nameplate_energy          :math:`E`
charging_eta              :math:`\eta_c`
discharging_eta           :math:`\eta_d`
degradation_rate          :math:`d`
initial_state_of_charge   :math:`SOC_{init}`
initial_energy_throughput :math:`ET_{init}`
dt                        :math:`\Delta t`      Used to convert power flows into energy holdup
elec_in                   :math:`I_{t}`
elec_out                  :math:`O_{t}`
state_of_charge           :math:`SOC_{t}`
energy_throughput         :math:`ET_{t}`
========================= ===================== ===========================================

Constraints
---------------
State of charge evolves with electricity in and out flows:

.. math:: SOC_t = SOC_{init} + \eta_c \times \Delta t \times I_t - \frac{\Delta t \times O_t}{\eta_d}

Energy throughput is accumulated over time:

.. math:: ET_t = ET_{init} + \Delta t \times \frac{I_t + O_t}{2}

Storage capacity is limited by degradation which increases with energy throughput:

.. math:: SOC_t \leq E - d \times ET_t


Battery Class
--------------

.. module:: dispatches.models.renewables_case.battery

.. autoclass:: BatteryStorage
  :members:
