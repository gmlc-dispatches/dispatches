.. _electricalsplitter:

Electrical Splitter
====================

The DISPATCHES Electrical Splitter Model represents operations where a single flow of electricity is split into
multiple flows using split fractions. This Electricity Splitter Model is similar to the IDAES Separator unit model
except that the inlets and outlets are electricity flows without any material properties.

Degrees of Freedom
------------------

The Electrical Splitter Model has :math:`(no. outlets - 1)` degrees of freedom.

Typical fixed variables are the split fractions.


Model Structure
---------------

The Electrical Splitter Model uses electricity flow balance to split the inlet stream into a number of outlet streams.
The model has a single inlet Port (named `electricity_in`) and a user-defined number of outlet Ports, which by default
are named `outlet_1_elec`, `outlet_2_elec`, etc. Outlets can also be created with custom names using an `outlet_list` in
the configuration.


Variables
--------------

The Electrical Splitter Model uses the following variables (:math:`o` indicates index by outlet):

======================= ===================== ===========================================
Variable Name           Symbol                       Notes
======================= ===================== ===========================================
electricity             :math:`I_{t}`         Inlet
outlet_1_elec, ...      :math:`O_{t, o}`      Outlets
======================= ===================== ===========================================

Constraints
---------------
Sum of `split_fraction` is 1:

.. math:: I_{t} =\sum_o O_{t, o}


Electrical Splitter Class
-------------------------

.. module:: dispatches.models.renewables_case.elec_splitter

.. autoclass:: ElectricalSplitter
  :members:

