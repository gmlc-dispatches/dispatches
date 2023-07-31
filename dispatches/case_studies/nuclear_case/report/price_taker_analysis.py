#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
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

# General python imports
import json
from importlib import resources
from pathlib import Path
import pandas as pd

# Pyomo imports
from pyomo.environ import (
    Var,
    value,
    NonNegativeReals,
    Constraint,
    Expression,
    Objective,
    Param,
    maximize,
    SolverFactory,
)

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.apps.grid_integration import MultiPeriodModel
import idaes.logger as idaeslog


H2_PROD_RATE = 20  # Rate of hydrogen production from the electrolyzer


def get_lmp_data(m, market="DA", filename="results"):
    """
    This function reads and appends LMP data to the model depending on the price-taker variant
    the user is interested in. There are four price-taker variants:
    V1: Uses day-ahead LMPs only
    V2: Uses real-time LMPs only
    V3: Uses max(day-ahead LMP, real-time LMP) at each hour
    V4: Two-step method: Uses both day-ahead and real-time LMPs

    Args:
        m: Pyomo model 
        market: Price-taker variant: V1 -> "DA", V2 -> "RT", V3 -> "Max-DA-RT", V4 -> "DA-RT"
        filename: csv file containing V1 results. This is needed for V4.

    Returns:
        None
    """
    with resources.path(
        "dispatches.case_studies.nuclear_case.report", 
        "rts_gmlc_15_500.csv"
    ) as p:
        path_to_file = Path(p).resolve()

    raw_data = pd.read_csv(path_to_file)

    if market == "DA-RT":
        price_all_1 = raw_data["LMP DA"].tolist()
        price_all_2 = raw_data["LMP"].tolist()

        # Read the optimal dispatch data obtained from Step 1
        dispatch_da_df = pd.read_csv(filename + "_schedule.csv")
        dispatch_da = dispatch_da_df["np_to_grid"].tolist()

        m.DISPATCH_DA = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(dispatch_da)},
            doc="DA dispatch [in MW]",
        )

        m.LMP_DA = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(price_all_1)},
            doc="Locational Marginal Prices [in $/MWh]",
        )

        m.LMP_RT = Param(
            m.set_period,
            initialize={t + 1: lmp for (t, lmp) in enumerate(price_all_2)},
            doc="Locational Marginal Prices [in $/MWh]",
        )

        return

    if market == "DA":
        price_all = raw_data["LMP DA"].tolist()

    elif market == "RT":
        price_all = raw_data["LMP"].tolist()

    elif market == "Max-DA-RT":
        price_all_1 = raw_data["LMP DA"].tolist()
        price_all_2 = raw_data["LMP"].tolist()
        price_all = [max([price_all_1[i], price_all_2[i]]) for i in range(len(price_all_1))]

    m.LMP = Param(
        m.set_period,
        initialize={t + 1: lmp for (t, lmp) in enumerate(price_all)},
        doc="Locational Marginal Prices [in $/MWh]",
    )


def build_ne_flowsheet(m):
    """
    Builds the flowsheet model for a baseload nuclear power plant integrated with 
    a PEM electrolyzer, a storage tank, and a hydrogen turbine. For the analysis in the report,
    the storage tank and the hydrogen turbine are not included by setting their capacities to zero.
    """
    m.fs = FlowsheetBlock(dynamic=False)

    # Declare variables
    m.fs.np_power = Var(within=NonNegativeReals, doc="Power produced by the nuclear plant (MW)")
    m.fs.np_to_grid = Var(within=NonNegativeReals, doc="Power from NP to the grid (MW)")
    m.fs.np_to_electrolyzer = Var(within=NonNegativeReals, doc="Power from NP to electrolyzer (MW)")

    m.fs.h2_production = Var(within=NonNegativeReals, doc="Hydrogen production rate (kg/hr)")

    m.fs.tank_holdup = Var(within=NonNegativeReals, doc="Hydrogen holdup in the tank (kg)")
    m.fs.tank_holdup_previous = Var(within=NonNegativeReals, doc="Hold at the beginning of the period (kg)")
    m.fs.h2_to_pipeline = Var(within=NonNegativeReals, doc="Hydrogen flowrate to the pipeline (kg/hr)")
    m.fs.h2_to_turbine = Var(within=NonNegativeReals, doc="Hydrogen flowrate to the turbine (kg/hr)")

    m.fs.h2_turbine_power = Var(within=NonNegativeReals, doc="Power production from H2 turbine (MW)")
    m.fs.net_power = Var(within=NonNegativeReals, doc="Net power to the grid (MW)")

    # Capacity of the nuclear power plant 121_NUCLEAR_1 in RTS GMLC dataset = 400 MW
    m.fs.np_power.fix(400)

    # Declare Constraints
    m.fs.np_power_balance = Constraint(
        expr=m.fs.np_power == m.fs.np_to_grid + m.fs.np_to_electrolyzer,
        doc="Power balance at the nuclear power plant",
    )
    # Compute the hydrogen production rate
    # NEL PEM design ~= 50.4 ~ 50 kW-hr/kg of hydrogen
    m.fs.calc_h2_production_rate = Constraint(
        expr=m.fs.h2_production == H2_PROD_RATE * m.fs.np_to_electrolyzer,
        doc="Computes the hydrogen production rate",
    )
    # Tank holdup calculations (Assuming a delta_t of 1 hr)
    m.fs.tank_mass_balance = Constraint(
        expr=m.fs.tank_holdup - m.fs.tank_holdup_previous ==
        (m.fs.h2_production - m.fs.h2_to_pipeline - m.fs.h2_to_turbine)
    )

    # Compute the power production via h2 turbine
    # For an air_h2_ratio of 10.76, (T, P) of h2 = (300 K, 1 atm),
    # delta_p across compressor and turbine 24.1 bar, the conversion
    # factor is 0.0125 MW-hr/kg hydrogen
    m.fs.calc_turbine_power = Constraint(
        expr=m.fs.h2_turbine_power == 0.0125 * m.fs.h2_to_turbine,
        doc="Computes the power production via h2 turbine",
    )
    # Power balance at the grid
    m.fs.grid_power_balance = Constraint(
        expr=m.fs.net_power == m.fs.np_to_grid + m.fs.h2_turbine_power
    )

    return m


def get_linking_variable_pairs(t1, t2):
    return [
        (t1.fs.tank_holdup, t2.fs.tank_holdup_previous)
    ]


def build_deterministic_model(demand_type="variable", demand=200):
    """
    Formulates the multiperiod price-taker problem for the nuclear case study

    Args:
        demand_type: "fixed" or "variable" demand for hydrogen
        demand: Hydrogen demand [kg/hr]
    """

    m = MultiPeriodModel(
        n_time_points=366 * 24,
        process_model_func=build_ne_flowsheet,
        linking_variable_func=get_linking_variable_pairs,
        use_stochastic_build=True,
        outlvl=idaeslog.WARNING,
    )

    # Declare first-stage variables
    m.pem_capacity = Var(within=NonNegativeReals, doc="Maximum capacity of the PEM electrolyzer (in MW)")
    m.tank_capacity = Var(within=NonNegativeReals, doc="Maximum holdup of the tank (in kg)")
    m.h2_turbine_capacity = Var(within=NonNegativeReals, doc="Maximum power output from the turbine (in MW)")

    @m.Constraint(m.set_period, doc="Power input to PEM cannot exceed its capacity")
    def pem_capacity_constraint(blk, t):
        return blk.period[t].fs.np_to_electrolyzer <= blk.pem_capacity

    @m.Constraint(m.set_period, doc="Tank holdup cannot exceed its capacity")
    def tank_capacity_constraint(blk, t):
        return blk.period[t].fs.tank_holdup <= blk.tank_capacity

    @m.Constraint(m.set_period, doc="Power output from turbine cannot exceed its capacity")
    def turbine_capacity_constraint(blk, t):
        return blk.period[t].fs.h2_turbine_power <= blk.h2_turbine_capacity

    for t in m.set_period:
        if demand_type == "fixed":
            m.period[t].fs.h2_to_pipeline.fix(demand)

        elif demand_type == "variable":
            m.period[t].fs.h2_to_pipeline.setub(demand)

    return m


def append_op_costs_revenue(
    m,
    lmp,
    h2_price,
    vom_npp=2.3,
    vom_pem=1.3,
    vom_turbine=4.25,
    lmp_da=None,
    dispatch_da=None,
):
    """
    Appends operating costs and revenue generated at each hour
    """
    # Total variable O&M cost
    m.fs.vom = Expression(
        expr=vom_pem * m.fs.np_to_electrolyzer + vom_turbine * m.fs.h2_turbine_power + vom_npp * m.fs.np_power
    )
    if lmp_da is None:
        m.fs.electricity_revenue = Expression(expr=lmp * m.fs.net_power)
    
    else:
        m.fs.electricity_revenue = Expression(
            expr=lmp_da * dispatch_da + lmp * (m.fs.net_power - dispatch_da)
        )

    m.fs.h2_revenue = Expression(expr=h2_price * m.fs.h2_to_pipeline)

    m.fs.net_cash_inflow = Expression(
        expr=m.fs.h2_revenue + m.fs.electricity_revenue - m.fs.vom
    )


def append_npv_calculations(
    m,
    plant_life=30,
    tax_rate=0.2,
    discount_rate=0.08,
    capex_pem=1630,
    capex_tank=29,
    capex_turbine=947,
    fom_pem=47.9,
    fom_turbine=7
):
    """
    Appends overall NPV calculations.
    """
    # Note: The capex and fom numbers for pem and turbine are in $/kW 
    # and the capex of the tank is in $/kWh
    # Note: LHV of hydrogen is 33.3 kWh/kg
    m.pem_capex = Expression(expr=capex_pem * 1000 * m.pem_capacity)
    m.tank_capex = Expression(expr=(capex_tank * 33.3) * m.tank_capacity)
    m.turbine_capex = Expression(expr=capex_turbine * 1000 * m.h2_turbine_capacity)
    m.pem_fom = Expression(expr=1000 * fom_pem * m.pem_capacity)
    m.turbine_fom = Expression(expr=1000 * fom_turbine * m.h2_turbine_capacity)

    m.capex = Expression(
        expr=m.pem_capex + m.tank_capex + m.turbine_capex,
        doc="Total capital cost (in USD)"
    )

    m.npp_fom = Expression(expr=120 * 1000 * 400)

    m.fixed_om_cost = Expression(
        expr=m.pem_fom + m.turbine_fom + m.npp_fom,
        doc="Fixed O&M Cost (in USD)"
    )

    # Variable O&M: PEM: $1.3/MWh and turbine: $4.25/MWh
    m.total_cash_inflow = Expression(
        expr=sum(m.period[t].fs.net_cash_inflow for t in m.set_period),
        doc="Computes the total cash inflow (in USD)"
    )

    m.depreciation = Expression(
        expr=m.capex / plant_life,
        doc="Straight-line depreciation with zero salvage value",
    )

    m.net_profit = Expression(
        expr=m.depreciation + (1 - tax_rate) * (m.total_cash_inflow - m.fixed_om_cost - m.depreciation)
    )

    # Factor for constant cash flow
    m.constant_cf_factor = (1 - (1 + discount_rate) ** (- plant_life)) / discount_rate


def append_objective_function(m):
    m.npv = Objective(
        expr=m.constant_cf_factor * m.net_profit - m.capex,
        sense=maximize
    )


def append_annualized_objective_function(m):
    m.npv = Objective(
        expr=m.net_profit - (1 / m.constant_cf_factor) * m.capex,
        sense=maximize
    )


def _write_results(m, filename="results"):
    # Create a directory to store NPV optimization results
    _filename = filename + "_schedule.csv"
    set_flowsheets = m.period[:].fs
    LMP = m.LMP if hasattr(m, "LMP") else m.LMP_DA

    results = {
        "LMP [$/MWh]": [LMP[t] for t in m.set_period],
        "np_to_grid": [value(fs.np_to_grid) for fs in set_flowsheets],
        "np_to_electrolyzer": [value(fs.np_to_electrolyzer) for fs in set_flowsheets],

        "tank_holdup_previous": [value(fs.tank_holdup_previous) for fs in set_flowsheets],
        "tank_holdup": [value(fs.tank_holdup) for fs in set_flowsheets],
        "h2_to_pipeline": [value(fs.h2_to_pipeline) for fs in set_flowsheets],
        "h2_to_turbine": [value(fs.h2_to_turbine) for fs in set_flowsheets],
        "h2_turbine_power": [value(fs.h2_turbine_power) for fs in set_flowsheets],

        "h2_revenue": [value(fs.h2_revenue) for fs in set_flowsheets],
        "electricity_revenue": [value(fs.electricity_revenue) for fs in set_flowsheets],
        # "electricity_cost": [value(fs.electricity_cost) for fs in set_flowsheets],
        "vom": [value(fs.vom) for fs in set_flowsheets],
        "net_cash_inflow": [value(fs.net_cash_inflow) for fs in set_flowsheets],
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(_filename)


def run_exhaustive_enumeration(pem_capex=400, market="DA"):
    """
    Performs sensitivity analysis with respect to the ratio of capacities of PEM and 
    electrolyzer and the selling price of hydrogen.
    """
    h2_price = [0.75, 1, 1.25, 1.5, 1.75, 2]
    pem_capacity = [i / 100 for i in range(5, 51, 5)]

    h2_demand = 400 * 20
    demand_type = "variable"
    vom_pem = 0

    solver = SolverFactory("gurobi")
    results = {
        "h2_price": h2_price, "pem_cap": pem_capacity, "solver_stat": {},
        "elec_rev": {}, "h2_rev": {}, "net_npv": {}, "net_profit": {},
        "pem_cap_factor": {},
    }

    for idx1, hp in enumerate(h2_price):
        for idx2, pc in enumerate(pem_capacity):
            m = build_deterministic_model(demand_type=demand_type, demand=h2_demand)

            # Set tank  and turbine capacity to zero to prevent building one
            m.period[1].fs.tank_holdup_previous.fix(0)
            m.tank_capacity.fix(0)
            m.h2_turbine_capacity.fix(0)

            get_lmp_data(m, market=market, filename="case_" + str(idx1) + "_" + str(idx2))

            if market in ["DA", "RT", "Max-DA-RT"]:
                for t in m.set_period:
                    append_op_costs_revenue(
                        m=m.period[t], lmp=m.LMP[t], h2_price=hp, vom_pem=vom_pem,
                    )

            elif market == "DA-RT":
                for t in m.set_period:
                    append_op_costs_revenue(
                        m=m.period[t], lmp=m.LMP_RT[t], h2_price=hp, vom_pem=vom_pem,
                        lmp_da=m.LMP_DA[t], dispatch_da=m.DISPATCH_DA[t],
                    )

            append_npv_calculations(m, capex_pem=pem_capex, fom_pem=0.03*pem_capex)
            append_annualized_objective_function(m)

            n_hours = len([t for t in m.period])

            print("Solving case number: ", (idx1, idx2))
            m.pem_capacity.fix(pc * 400)
            soln = solver.solve(m)

            elec_rev = sum(value(m.period[t].fs.electricity_revenue) for t in m.set_period)
            h2_rev = sum(value(m.period[t].fs.h2_revenue) for t in m.set_period)
            pem_cap_factor = (
                sum(value(m.period[t].fs.np_to_electrolyzer) for t in m.set_period) /
                (pc * 400 * n_hours)
            )

            results["elec_rev"][str(idx1) + str(idx2)] = elec_rev / 1e6
            results["h2_rev"][str(idx1) + str(idx2)] = h2_rev / 1e6
            results["net_npv"][str(idx1) + str(idx2)] = value(m.npv) / 1e6
            results["net_profit"][str(idx1) + str(idx2)] = value(m.net_profit) / 1e6
            results["solver_stat"][str(idx1) + str(idx2)] = str(soln.solver.termination_condition)
            results["pem_cap_factor"][str(idx1) + str(idx2)] = pem_cap_factor

            _write_results(m, filename="case_" + str(idx1) + "_" + str(idx2))

    results["capex"] = value(m.capex) / 1e6
    results["fom"] = value(m.fixed_om_cost) / 1e6

    with open("price_taker_pem_" + str(pem_capex) + "_" + market + ".json", "w") as fp:
        json.dump(results, fp, indent=4)


if __name__ == '__main__':
    run_exhaustive_enumeration(pem_capex=1200, market="RT")