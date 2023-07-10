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

import json


def ne_traditional_tea(
    npp_pem_ratio=0.5, 
    pem_cap_factor=0.75,
    h2_selling_price=0.75,
    pem_capex=1200,
    vom_npp=2.3,
):
    """
    Performs traditional techno-economic analysis for hybridizing an existing nuclear
    generator with an electrolysis unit.

    Args:
        npp_pem_ratio: Ratio of capacities of PEM and nuclear generator [-]
        pem_cap_factor: Capacity factor of the electrolyzer [-]
        h2_selling_price: Selling price of hydrogen [$/kg]
        pem_capex: Overnight capital cost of the electrolyzer [$/kW]
        vom_npp: Variable O&M cost of the nuclear generator [$/MWh]

    Returns:
        npv: Annualized net present value
        elec_revenue: Annual revenue from the electricity market
        h2_revenue: Annual revenue from the hydrogen market
    """

    # Declare parameter values
    npp_capacity = 400              # Capacity of the baseload nuclear generator [MW]
    avg_lmp = 22.09341              # Average day-ahead LMP at the bus Attlee [$/MWh]
    h2_prod_rate = 20               # Rate of hydrogen production [kg/MWh]
    num_hours = 8784                # Number of hours in a year
    discount_rate = 0.08            # Discount rate [-]
    plant_life = 30                 # Lifetime of the electrolyzer [years]
    tax_rate = 0.2                  # Corporate tax rate [-]
    vom_pem = 0                     # Variable O&M cost of PEM [$/MWh]
    fom_npp = 120 * 1000            # Fixed O&M cost nuclear generator [$/MW-year]
    _pem_capex = pem_capex * 1000   # CAPEX of PEM [$/MW]
    fom_pem = 0.03 * _pem_capex     # Fixed O&M cost of PEM [$/MW-year]
    annualization_factor = (1 - (1 + discount_rate) ** (- plant_life)) / discount_rate
    pem_capacity = npp_capacity * npp_pem_ratio

    # Traditional techno-economic analysis
    h2_produced = (pem_capacity * h2_prod_rate) * num_hours * pem_cap_factor
    electricity_sold = (npp_capacity * num_hours) - (pem_capacity * num_hours * pem_cap_factor)
    h2_revenue = h2_produced * h2_selling_price
    elec_revenue = electricity_sold * avg_lmp
    total_vom = npp_capacity * num_hours * vom_npp + pem_capacity * num_hours * vom_pem
    capex = _pem_capex * pem_capacity
    total_fom = fom_pem * pem_capacity + fom_npp * npp_capacity
    depreciation = capex / plant_life
    _tax = max(0, tax_rate * (h2_revenue + elec_revenue - total_vom - total_fom - depreciation))
    net_profit = h2_revenue + elec_revenue - total_vom - total_fom - _tax

    npv = net_profit - (1 / annualization_factor) * capex

    return npv, elec_revenue, h2_revenue


def run_exhaustive_enumeration(pem_capex=400):
    """
    Performs sensitivity analysis with respect to the selling price of 
    hydrogen and the ratio of capacities and saves the results in a json file
    """

    h2_price = [0.75, 1, 1.25, 1.5, 1.75, 2]
    pem_capacity = [i / 100 for i in range(5, 51, 5)]

    results = {
        "h2_price": h2_price, "pem_cap": pem_capacity, "solver_stat": {},
        "elec_rev": {}, "h2_rev": {}, "net_npv": {}, "net_profit": {},
        "pem_cap_factor": {},
    }

    for idx1, hp in enumerate(h2_price):
        for idx2, pc in enumerate(pem_capacity):

            npv, elec_rev, h2_rev = ne_traditional_tea(
                npp_pem_ratio=pc,
                h2_selling_price=hp,
                pem_capex=pem_capex,
            )

            results["elec_rev"][str(idx1) + str(idx2)] = elec_rev / 1e6
            results["h2_rev"][str(idx1) + str(idx2)] = h2_rev / 1e6
            results["net_npv"][str(idx1) + str(idx2)] = npv / 1e6
            results["pem_cap_factor"][str(idx1) + str(idx2)] = 0.5

    with open("traditional_tea_pem_" + str(pem_capex) + ".json", "w") as fp:
        json.dump(results, fp, indent=4)


if __name__ == "__main__":
    run_exhaustive_enumeration(pem_capex=400)
