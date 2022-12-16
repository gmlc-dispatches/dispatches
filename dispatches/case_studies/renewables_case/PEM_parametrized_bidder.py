from idaes.apps.grid_integration.bidder import *
from dispatches.case_studies.renewables_h2_case.parametrized_bidder import PerfectForecaster, ParametrizedBidder

class PEMParametrizedBidder(ParametrizedBidder):

    """
    Template class for bidders that use fixed parameters.
    """

    def __init__(
        self,
        bidding_model_object,
        day_ahead_horizon,
        real_time_horizon,
        n_scenario,
        solver,
        forecaster,
        pem_marginal_cost,
        pem_mw
    ):
        super().__init__(bidding_model_object,
                         day_ahead_horizon,
                         real_time_horizon,
                         n_scenario,
                         solver,
                         forecaster)
        self.wind_marginal_cost = 0
        self.wind_mw = self.bidding_model_object._wind_pmax_mw
        self.pem_marginal_cost = pem_marginal_cost
        self.pem_mw = pem_mw

    def compute_day_ahead_bids(self, date, hour=0):
        gen = self.generator
        forecast = self.forecaster.forecast_day_ahead_capacity_factor(date, hour, gen, self.day_ahead_horizon)

        full_bids = {}

        for t_idx in range(self.day_ahead_horizon):
            da_wind = forecast[t_idx] * self.wind_mw
            grid_wind = max(0, da_wind - self.pem_mw)
            bids = [(0, 0), (grid_wind, 0), (da_wind, self.pem_marginal_cost)]
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
            full_bids[t][gen]["p_max"] = da_wind
            full_bids[t][gen]["startup_capacity"] = da_wind
            full_bids[t][gen]["shutdown_capacity"] = da_wind

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
            grid_wind = max(0, rt_wind - self.pem_mw)
            bids = [(0, 0), (grid_wind, 0), (rt_wind, self.pem_marginal_cost)]

            t = t_idx + hour
            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = convert_marginal_costs_to_actual_costs(bids)
            full_bids[t][gen]["p_min"] = 0
            full_bids[t][gen]["p_max"] = rt_wind
            full_bids[t][gen]["startup_capacity"] = rt_wind
            full_bids[t][gen]["shutdown_capacity"] = rt_wind

        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids
