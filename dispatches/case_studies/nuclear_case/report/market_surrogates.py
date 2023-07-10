#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

__author__ = "Radhakrishna Tumbalam Gooty"

# This code requires OMLT v1.0

import os
import json
from tensorflow import keras

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import (
    ConcreteModel,
    Var, 
    NonNegativeReals, 
    Expression, 
    Constraint,
    Objective, 
)

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver

import omlt  # omlt can encode the neural networks in Pyomo
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import load_keras_sequential

PEM_CAPEX = 1200
LIFETIME = 30
TAX_RATE = 0.2                  # Corporate tax rate
DISC_RATE = 0.08                # Discount rate
r_def = 1 / (1 + DISC_RATE)
# Annualization factor for CAPEX of the system
ANN_FACTOR = (1/r_def) * ((1-r_def) / (1-r_def ** LIFETIME)) 

NP_CAPACITY = 400               # Capacity of the nuclear power plant (in MW)
H2_PROD_RATE = (1000 / 50)      # Hydrogen production rate (in kg/MW-h)
NUM_HOURS = 8784                # Number of hours in a year

NPP_FOM = 13.7  # Normalized FOM = (120,000 / 8760) = $13.7 per MWh
NPP_VOM = 2.3   # Using a VOM cost of $2.3 per MWh for the nuclear power plant
PEM_FOM = 5.47  # Normalized FOM = (47,900 / 8760) = $5.47 per MWh
PEM_VOM = 0.0   # Using a VOM cost of $1.3 per MWh for the PEM electrolyzer

KW_TO_MW = 1e-3
HOURS_TO_S = 3600
MW_H2 = 2.016e-3

# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(), "nn_steady_state")

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "revenue_input_output_scaling_data.json"), 'rb') as f:
    revenue_data = json.load(f)

input_bounds_revenue = {
    i: (revenue_data['xmin'][i], revenue_data['xmax'][i]) 
    for i in range(len(revenue_data['xmin']))
}
scaling_object_revenue = omlt.OffsetScaling(
    offset_inputs=revenue_data["xm_inputs"],
    factor_inputs=revenue_data["xstd_inputs"],
    offset_outputs=revenue_data["y_mean"],
    factor_outputs=revenue_data["y_std"],
)

nn_revenue = keras.models.load_model(os.path.join(surrogate_dir, "keras_revenue"))
revenue_defn = load_keras_sequential(nn_revenue, scaling_object_revenue, input_bounds_revenue)

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "cap_factor_input_output_scaling_data.json"), 'rb') as f:
    cap_factor_data = json.load(f)

input_bounds_cap_factor = {
    i: (cap_factor_data['xmin'][i], cap_factor_data['xmax'][i]) 
    for i in range(len(cap_factor_data['xmin']))
}
scaling_object_cap_factor = omlt.OffsetScaling(
    offset_inputs=cap_factor_data["xm_inputs"],
    factor_inputs=cap_factor_data["xstd_inputs"],
    offset_outputs=cap_factor_data["y_mean"],
    factor_outputs=cap_factor_data["y_std"],
)

nn_cap_factor = keras.models.load_model(os.path.join(surrogate_dir, "keras_cap_factor"))
cap_factor_defn = load_keras_sequential(
    nn_cap_factor, 
    scaling_object_cap_factor, 
    input_bounds_cap_factor,
)
# ====================================================================================================


def conceptual_design_ss_NE(reserve=10, max_lmp=500, H2_SELLING_PRICE=2):
    """
    Formulates the conceptual design problem for the nuclear case study

    Args:
        reserve: Percentage reserves
        max_lmp: Shortfall price
        H2_SELLING_PRICE: Selling price of hydrogen

    Returns:
        m: Optimization model embedding market surrogates
    """
    
    m = ConcreteModel()

    # Define a variable for the PEM capacity
    m.pem_capacity = Var(
        within=NonNegativeReals,
        doc="Capacity of the PEM electrolyzer [in MW]",
    )

    # Inputs to the frequency surrogates: PEM capacity/NP capacity, threshold price, reserve, max_lmp
    m.pem_np_cap_ratio = Var(
        within=NonNegativeReals,
        bounds=(0.05, 0.5),
        initialize=0.25,
        doc="Ratio of capacities of PEM and nuclear power plant",
    )
    m.threshold_price = Var(
        within=NonNegativeReals,
        initialize=20,
        doc="Threshold LMP below which selling H2 is more profitable",
    )
    m.reserve = Var(
        within=NonNegativeReals,
        doc="Percentage of reserves",
    )
    m.max_lmp = Var(
        within=NonNegativeReals,
        doc="Maximum LMP",
    )
    m.reserve.fix(reserve)
    m.max_lmp.fix(max_lmp)  

    # Relation between PEM capacity and pem_np_cap_ratio
    m.pem_capacity_definition = Constraint(
        expr=m.pem_capacity == m.pem_np_cap_ratio * NP_CAPACITY
    )
    # Threshold price calculation
    m.threshold_price_definition = Constraint(
        expr=m.threshold_price == H2_PROD_RATE * H2_SELLING_PRICE
    )

    # Add NN surrogate for revenue to the model
    m.nn_revenue = omlt.OmltBlock()
    m.nn_revenue.build_formulation(FullSpaceNNFormulation(revenue_defn))

    # Add NN surrogate for capacity factor to the model
    m.nn_cap_factor = omlt.OmltBlock()
    m.nn_cap_factor.build_formulation(FullSpaceNNFormulation(cap_factor_defn))

    inputs = [m.threshold_price, m.pem_np_cap_ratio, m.reserve, m.max_lmp]  

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_cap_factor(blk, i):
        return inputs[i] == blk.nn_cap_factor.inputs[i]

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_revenue(blk, i):
        return inputs[i] == blk.nn_revenue.inputs[i]

    m.electricity_revenue = Var(within=NonNegativeReals, initialize=1e9)
    m.revenue_surrogate = Constraint(expr=m.electricity_revenue == m.nn_revenue.outputs[0])

    m.npp_capacity_factor = Var(within=NonNegativeReals, initialize=0.75)
    m.cap_factor_surrogate = Constraint(expr=m.npp_capacity_factor == m.nn_cap_factor.outputs[0])

    # Define flowsheet variables
    m.net_energy_to_pem = Var(
        within=NonNegativeReals, 
        initialize=(0.25 * NP_CAPACITY * NUM_HOURS),
    )
    m.net_h2_production = Var(
        within=NonNegativeReals, 
        initialize=(0.25 * NP_CAPACITY * NUM_HOURS * H2_PROD_RATE),
    )

    # Calculate net power to PEM
    m.pem_energy_calculation = Constraint(
        expr=m.net_energy_to_pem == (1 - m.npp_capacity_factor) * NP_CAPACITY * NUM_HOURS
    )

    # Calculate net hydrogen production
    m.net_h2_prod_calculation = Constraint(
        expr=m.net_h2_production == m.net_energy_to_pem * H2_PROD_RATE
    )

    # Calculate net hydrogen revenue
    m.h2_revenue = Expression(expr=H2_SELLING_PRICE * m.net_h2_production)

    assert degrees_of_freedom(m) == 1

    m.total_operating_cost = Expression(
        expr=366 * 24 * (NP_CAPACITY * NPP_VOM + m.net_energy_to_pem * PEM_VOM)
    )

    m.pem_cap_cost = Expression(expr=ANN_FACTOR * PEM_CAPEX * 1000 * m.pem_capacity)
    m.depreciation = Expression(expr=((PEM_CAPEX * 1000) / LIFETIME) * m.pem_capacity)
    m.pem_fom = Expression(expr=0.03 * (PEM_CAPEX * 1000) * m.pem_capacity)

    # Assuming FOM of NPP to be $120/kW
    m.npp_fom = Expression(expr=120 * 1000 * NP_CAPACITY)
    m.net_profit = Expression(
        expr=m.depreciation + (1 - TAX_RATE) * (
            m.electricity_revenue + m.h2_revenue - m.total_operating_cost - m.pem_fom 
            - m.npp_fom - m.depreciation)
    )

    # set objective functions in $
    m.obj = Objective(expr=m.pem_cap_cost - m.net_profit)
    
    return m


def run_exhaustive_enumeration(reserve, max_lmp):
    h2_price = [0.75, 1, 1.25, 1.5, 1.75, 2]
    pem_cap = [i / 100 for i in range(5, 51, 5)]

    solver = get_solver()
    results = {
        "h2_price": h2_price, "pem_cap": pem_cap, "pem_cap_factor": {},
        "elec_rev": {}, "h2_rev": {}, "net_npv": {}, "solver_stat": {},
    }

    solver = get_solver()
    
    remaining_cases = []

    for idx1, hp in enumerate(h2_price):
        m = conceptual_design_ss_NE(reserve=reserve, max_lmp=max_lmp, H2_SELLING_PRICE=hp)

        m.pem_capacity.fix(pem_cap[0] * NP_CAPACITY)
        soln = solver.solve(m)

        for idx2, pc in enumerate(pem_cap):
            # The main purpose of this loop is to find a feasible solution
            m.pem_capacity.fix(pc * NP_CAPACITY)
            soln = solver.solve(m)

            if str(soln.solver.termination_condition) == "optimal":
                break

        unsolved_cases = []

        for idx2, pc in enumerate(pem_cap):
            # This loop actually solves the problem and records the unsolved cases
            print("Solving case: ", (idx1, idx2))
            m.pem_capacity.fix(pc * NP_CAPACITY)

            soln = solver.solve(m)

            if str(soln.solver.termination_condition) == "infeasible":
                unsolved_cases.append((idx2, pc))
                continue

            results["elec_rev"][str(idx1) +  str(idx2)] = m.electricity_revenue.value / 1e6
            results["h2_rev"][str(idx1) +  str(idx2)] = m.h2_revenue.expr() / 1e6
            results["net_npv"][str(idx1) +  str(idx2)] = -m.obj.expr() / 1e6
            results["solver_stat"][str(idx1) +  str(idx2)] = str(soln.solver.termination_condition)
            results["pem_cap_factor"][str(idx1) + str(idx2)] = \
                m.net_energy_to_pem.value / (m.pem_capacity.value * NUM_HOURS)
            
            if idx1 == 0 and idx2 == 1:
                # Attempts to solve a few cases which failed earlier
                idx2 = 0
                pc = 0.05

                print("Solving case: ", (idx1, idx2))
                m.pem_capacity.fix(pc * NP_CAPACITY)

                soln = solver.solve(m)

                if str(soln.solver.termination_condition) == "infeasible":
                    continue

                results["elec_rev"][str(idx1) +  str(idx2)] = m.electricity_revenue.value / 1e6
                results["h2_rev"][str(idx1) +  str(idx2)] = m.h2_revenue.expr() / 1e6
                results["net_npv"][str(idx1) +  str(idx2)] = -m.obj.expr() / 1e6
                results["solver_stat"][str(idx1) +  str(idx2)] = str(soln.solver.termination_condition)
                results["pem_cap_factor"][str(idx1) + str(idx2)] = \
                    m.net_energy_to_pem.value / (m.pem_capacity.value * NUM_HOURS)

        for idx2, pc in unsolved_cases:
            print("Solving case: ", (idx1, idx2))
            m.pem_capacity.fix(pc * NP_CAPACITY)

            soln = solver.solve(m)

            if str(soln.solver.termination_condition) == "infeasible":
                remaining_cases.append((idx2, pc))
                continue

            results["elec_rev"][str(idx1) +  str(idx2)] = m.electricity_revenue.value / 1e6
            results["h2_rev"][str(idx1) +  str(idx2)] = m.h2_revenue.expr() / 1e6
            results["net_npv"][str(idx1) +  str(idx2)] = -m.obj.expr() / 1e6
            results["solver_stat"][str(idx1) +  str(idx2)] = str(soln.solver.termination_condition)
            results["pem_cap_factor"][str(idx1) + str(idx2)] = \
                m.net_energy_to_pem.value / (m.pem_capacity.value * NUM_HOURS)

    print(remaining_cases)

    with open("ss_results_" + str(reserve) + "_" + str(max_lmp) + 
              "_" + str(PEM_CAPEX) + ".json", "w") as fp:
        json.dump(results, fp, indent=4)


if __name__ == '__main__':
    run_exhaustive_enumeration(reserve=15, max_lmp=500)
