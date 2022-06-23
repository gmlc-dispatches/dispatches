Wind-Battery-Hydrogen Hybrid Conceptual Design
==============================================

The Renewable Case Study looks at hybridizing a Wind plant with battery storage and hydrogen production for both storage and sales.

The simplest wind-battery-hydrogen plant is modeled as a wind plant with battery and a PEM electrolyzer in `dispatches/renewables_case/wind_battery_PEM_LMP.py`. 
The electricity flow from the wind is split among charging the battery, producing hydrogen via the PEM and selling directly to the grid.
This model is built using the :ref:`windpower`, :ref:`battery`, :ref:`pemelectrolyzer` and :ref:`electricalsplitter` unit models. 

The hydrogen can be used for power production as well, by adding the :ref:`hydrogentank` and :ref:`hydrogenturbine` unit models to the flowsheet.
The conceptual design optimization code for this model is found at `dispatches/renewables_case/wind_battery_PEM_tank_turbine_LMP.py`.
For an example Jupyter Notebook, please see `dispatches/renewables_case/ConceptualDesignOptimization.pynb`.
