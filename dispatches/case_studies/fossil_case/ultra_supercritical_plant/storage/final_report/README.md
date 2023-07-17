# Fossil Energy Case Study: Integrated Ultra-supercritical Power Plant (USCPP)

This directroy includes all the models used for the solution of the two-step approach for the simultaneous design and operation of an ultra-supercritical power plant integrated with a thermal energy storage system. The two steps are:
Step 1. Generalized Disjunctive Programming (GDP) implementation for the optimal design of an integrated USCPP with a charge and discharge process.
Step 2. Multi-period model under price-taker assumption to determine the influence of electricity prices in the design and operation of integrated USCPP.

## Models

- uscpp_charge_superstructure.py: Solves the charge GDP superstructure in Step 1
- uscpp_discharge_superstructure.py: Solves the discharge GDP superstructure in Step 1 
- fixed_pricetaker.py: Solves the multi-period model under price-taker assumption in Step 2 at given fixed design capacity variables for the storage system.
- simultaneous_pricetaker.py: Solves the multi-period model under price-taker assumption in Step 2 for one and seven representative days. To change the number of days in the model, change line \esr{@Naresh, please add the line number here of the number_of_days} in the model.

## Software and Hardware
All the models are run on a Dell Precision 5560 laptop with 11th Gen Intel(R) Core(TM) i9-1195oH @ 2.60GHx, having 32.0 GB RAM and running Windows 11 Pro. The versions for the DISPATCHES platforms and solvers used are given below:
- DISPATCHES: Version (@Naresh, add version here)
- GDPopt: Version 22.5.13
- Gurobi: Version 10.0.1
- IPOPT: Version 3.13.2

## Output
The output obtained for all the models is included in the directory `output_files`. The output file for each model has the same source name from the model file, followed by the work `output`.  