# Renewable Energy Case Study: Wind + PEM and Wind + Battery Plants

This directory contains the files required for the renewable energy case studies, which use flowsheets in which any of the following 
technologies (modeled as unit models) may be present: Wind, PV, Battery, PEM Electrolysis, Hydrogen Storage Tank and Hydrogen Turbine.
The RE case studies focused on a Wind + PEM plant and a Wind + Battery plant, while the the industry-partnership case study looked at a PV + Battery + PEM + Hydrogen.
Each of these three case studies has different approaches to modeling prices or the grid.

## Wind + PEM case:

1. Price-taker Design Optimization: `dispatches/case_studies/renewables_case/run_pricetaker_wind_PEM.py`
2. Market Surrogate Design Optimization: `dispatches/case_studies/renewables_case/RE_surrogate_optimization_steadystate.py` 
3. Double Loop Simulation for Validation: `dispatches/case_studies/renewables_case/run_double_loop_PEM.py`
4. Market Surrogate Design and Validation Plotting: `dispatches/case_studies/renewables_case/SurrogateDesignResults.ipynb`

## Wind + Battery case:
1. Price-taker Design Optimization: `dispatches/case_studies/renewables_case/run_pricetaker_battery_ratio_size.py`
2. Double Loop Simulation: `dispatches/case_studies/renewables_case/run_double_loop_battery.py`
3. Parametrized Bidder Double Loop Simulation: `dispatches/case_studies/renewables_case/run_double_loop_battery_parametrized.py`

## PV + Battery + Hydrogen case:
1. Price-taker Design Optimization: `dispatches/case_studies/renewables_case/solar_battery_hydrogen.py`

## Software and Hardware
All the models are run on a Red Hat Enterprise Linux Server version 7.9 (Maipo). The versions for the solvers used are given below:
- Xpress: Version 8.13.0
- IPOPT: Version 3.13.2
