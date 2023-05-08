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
import numpy as np
from idaes.apps.grid_integration.bidder import convert_marginal_costs_to_actual_costs, tx_utils
from dispatches.workflow.parametrized_bidder import ParametrizedBidder


class FixedParametrizedBidder(ParametrizedBidder):

    """
    Template class for bidders that use fixed parameters.
    """

    def __init__(
        self,
        bidding_model_object,
        day_ahead_horizon,
        real_time_horizon,
        solver,
        forecaster,
        storage_marginal_cost,
        storage_mw
    ):
        super().__init__(bidding_model_object,
                         day_ahead_horizon,
                         real_time_horizon,
                         solver,
                         forecaster)
        self.wind_marginal_cost = 0
        self.wind_mw = self.bidding_model_object._wind_pmax_mw
        self.storage_marginal_cost = storage_marginal_cost
        self.storage_mw = storage_mw

    def compute_day_ahead_bids(self, date, hour=0):
        gen = self.generator
        forecast = self.forecaster.forecast_day_ahead_capacity_factor(date, hour, gen, self.day_ahead_horizon)

        full_bids = {}

        for t_idx in range(self.day_ahead_horizon):
            da_wind = forecast[t_idx] * self.wind_mw
            p_max = max(da_wind, self.storage_mw)
            bids = [(0, 0), (max(0, da_wind - self.storage_mw), 0), (p_max, self.storage_marginal_cost)]
            cost_curve = convert_marginal_costs_to_actual_costs(bids)

            temp_curve = {
                    "data_type": "cost_curve",
                    "cost_curve_type": "piecewise",
                    "values": cost_curve,
            }
            tx_utils.validate_and_clean_cost_curve(
                curve=temp_curve,
                curve_type="cost_curve",
                p_min=0,
                p_max=max([p[0] for p in cost_curve]),
                gen_name=gen,
                t=t_idx,
            )

            t = t_idx + hour
            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = cost_curve
            full_bids[t][gen]["p_min"] = 0
            full_bids[t][gen]["p_max"] = p_max
            full_bids[t][gen]["startup_capacity"] = p_max
            full_bids[t][gen]["shutdown_capacity"] = p_max

        self._record_bids(full_bids, date, hour, Market="Day-ahead")
        return full_bids

    def compute_real_time_bids(
        self, date, hour, realized_day_ahead_prices, realized_day_ahead_dispatches
    ):
        gen = self.generator
        forecast = self.forecaster.forecast_real_time_capacity_factor(date, hour, gen, self.day_ahead_horizon)
        
        full_bids = {}

        for t_idx in range(self.real_time_horizon):
            rt_wind = forecast[t_idx] * self.wind_mw
            p_max = max(rt_wind, self.storage_mw)
            bids = [(0, 0),  (max(0, rt_wind - self.storage_mw), 0), (p_max, self.storage_marginal_cost)]

            t = t_idx + hour
            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = convert_marginal_costs_to_actual_costs(bids)
            full_bids[t][gen]["p_min"] = 0
            full_bids[t][gen]["p_max"] = p_max
            full_bids[t][gen]["startup_capacity"] = p_max
            full_bids[t][gen]["shutdown_capacity"] = p_max

        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids