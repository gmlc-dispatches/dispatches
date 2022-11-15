#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
from idaes.apps.grid_integration.bidder import *
from idaes.apps.grid_integration.forecaster import AbstractPrescientPriceForecaster

class PerfectForecaster(AbstractPrescientPriceForecaster):

    def __init__(self, data_path_or_df):
        if isinstance(data_path_or_df, str):
            self.data = pd.read_csv(data_path_or_df, index_col="Datetime", parse_dates=True)
        elif isinstance(data_path_or_df, pd.DataFrame):
            self.data = data_path_or_df
        else:
            raise ValueError

    def __getitem__(self, index):
        return self.data[index]

    def fetch_hourly_stats_from_prescient(self, prescient_hourly_stats):
        pass

    def fetch_day_ahead_stats_from_prescient(self, uc_date, uc_hour, day_ahead_result):
        pass

    def forecast_day_ahead_and_real_time_prices(self, date, hour, bus, horizon, _):
        rt_forecast = self.forecast_real_time_prices(
            date, hour, bus, horizon, _
        )
        da_forecast = self.forecast_day_ahead_prices(
            date, hour, bus, horizon, _
        )
        return da_forecast, rt_forecast

    def forecast_day_ahead_prices(self, date, hour, bus, horizon, _):
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.data[self.data.index >= datetime_index].head(horizon)
        return forecast[f'{bus}-DALMP'].values

    def forecast_real_time_prices(self, date, hour, bus, horizon, _):
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.data[self.data.index >= datetime_index].head(horizon)
        return forecast[f'{bus}-RTLMP'].values

    def forecast_day_ahead_capacity_factor(self, date, hour, gen, horizon):
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.data[self.data.index >= datetime_index].head(horizon)
        return forecast[f'{gen}-DACF'].values

    def forecast_real_time_capacity_factor(self, date, hour, gen, horizon):
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.data[self.data.index >= datetime_index].head(horizon)
        return forecast[f'{gen}-RTCF'].values

class ParametrizedBidder(StochasticProgramBidder):

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
    ):
        super().__init__(bidding_model_object,
                         day_ahead_horizon,
                         real_time_horizon,
                         n_scenario,
                         solver,
                         forecaster,
                         real_time_underbid_penalty=500)
        self.battery_marginal_cost = 25
        self.battery_capacity_ratio = 0.4

    def formulate_DA_bidding_problem(self):
        pass

    def formulate_RT_bidding_problem(self):
        pass

    def assemble_bid(self, wind_forecast_energy, hour):
        full_bids = {}
        gen = self.generator

        for t_idx in range(self.day_ahead_horizon):
            power = wind_forecast_energy[t_idx] * (1.0 - self.battery_capacity_ratio)
            max_power = wind_forecast_energy[t_idx]
            t = t_idx + hour
            bids = [(power, 0), (max_power, self.battery_marginal_cost)]
            bid_curve = convert_marginal_costs_to_actual_costs(bids)

            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = bid_curve
            full_bids[t][gen]["p_min"] = power
            full_bids[t][gen]["p_max"] = power
            full_bids[t][gen]["startup_capacity"] = power
            full_bids[t][gen]["shutdown_capacity"] = power
        return full_bids

    def compute_day_ahead_bids(self, date, hour=0):
        forecast = self.forecaster.forecast_day_ahead_capacity_factor(date, hour, self.generator, self.day_ahead_horizon)
        da_wind = forecast * self.bidding_model_object._wind_pmax_mw
        full_bids = self.assemble_bid(da_wind, hour)
        self._record_bids(full_bids, date, hour, Market="Day-ahead")
        return full_bids

    def compute_real_time_bids(
        self, date, hour, realized_day_ahead_prices, realized_day_ahead_dispatches, tracker_profile
    ):
        forecast = self.forecaster.forecast_real_time_capacity_factor(date, hour, self.generator, self.day_ahead_horizon)
        rt_wind = forecast * self.bidding_model_object._wind_pmax_mw
        full_bids = self.assemble_bid(rt_wind, hour)
        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids

    def update_real_time_model(self, **kwargs):
        pass

    def update_day_ahead_model(self, **kwargs):
        pass

    def _record_bids(self, bids, date, hour, **kwargs):
        df_list = []
        for t in bids:
            for gen in bids[t]:

                result_dict = {}
                result_dict["Generator"] = gen
                result_dict["Date"] = date
                result_dict["Hour"] = t

                for k, v in kwargs.items():
                    result_dict[k] = v

                pair_cnt = len(bids[t][gen]["p_cost"])

                for idx, (power, cost) in enumerate(bids[t][gen]["p_cost"]):
                    result_dict[f"Power {idx} [MW]"] = power
                    result_dict[f"Cost {idx} [$]"] = cost

                # place holder, in case different len of bids
                while pair_cnt < self.n_scenario:
                    result_dict[f"Power {pair_cnt} [MW]"] = None
                    result_dict[f"Cost {pair_cnt} [$]"] = None

                    pair_cnt += 1

                result_df = pd.DataFrame.from_dict(result_dict, orient="index")
                df_list.append(result_df.T)

        # save the result to object property
        # wait to be written when simulation ends
        self.bids_result_list.append(pd.concat(df_list))

        return

    def write_results(self, path):
        """
        This methods writes the saved operation stats into an csv file.

        Arguments:
            path: the path to write the results.

        Return:
            None
        """

        print("")
        print("Saving bidding results to disk...")
        pd.concat(self.bids_result_list).to_csv(
            os.path.join(path, "bidder_detail.csv"), index=False
        )
        return


class VaryingParametrizedBidder(ParametrizedBidder):

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
    ):
        super().__init__(bidding_model_object,
                         day_ahead_horizon,
                         real_time_horizon,
                         n_scenario,
                         solver,
                         forecaster)
        self.da_cf_to_reserve = 0.8
        self.da_rt_price_threshold = 0.9

    def compute_day_ahead_bids(self, date, hour=0):
        gen = self.generator
        forecast = self.forecaster.forecast_day_ahead_capacity_factor(date, hour, gen, self.day_ahead_horizon)
        da_wind = forecast * self.bidding_model_object._wind_pmax_mw

        full_bids = {}

        for t_idx in range(self.day_ahead_horizon):
            power = da_wind[t_idx] * self.da_cf_to_reserve
            t = t_idx + hour

            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = [(power, 0)]
            full_bids[t][gen]["p_min"] = power
            full_bids[t][gen]["p_max"] = power
            full_bids[t][gen]["startup_capacity"] = power
            full_bids[t][gen]["shutdown_capacity"] = power

        self._record_bids(full_bids, date, hour, Market="Day-ahead")
        return full_bids

    def compute_real_time_bids(
        self, date, hour, realized_day_ahead_prices, realized_day_ahead_dispatches, tracker_profile
    ):
        gen = self.generator
        battery_avail_mwh = tracker_profile['realized_soc'][0] / self.real_time_horizon * 1e-3
        forecast = self.forecaster.forecast_real_time_capacity_factor(date, hour, gen, self.day_ahead_horizon)
        rt_wind = forecast * self.bidding_model_object._wind_pmax_mw

        rt_price = [i / self.da_rt_price_threshold for i in realized_day_ahead_prices]

        full_bids = {}

        for t_idx in range(self.real_time_horizon):
            power = rt_wind[t_idx] + battery_avail_mwh

            t = t_idx + hour

            full_bids[t] = {}
            full_bids[t][gen] = {}
            if power > realized_day_ahead_dispatches[t % 24]:
                bids = [(realized_day_ahead_dispatches[t % 24], 0), (power, rt_price[t_idx])]
                full_bids[t][gen]["p_cost"] = convert_marginal_costs_to_actual_costs(bids)
                p_min = realized_day_ahead_dispatches[t % 24]
            else:
                full_bids[t][gen]["p_cost"] = [(power, 0)]
                p_min = power
            full_bids[t][gen]["p_min"] = p_min
            full_bids[t][gen]["p_max"] = power
            full_bids[t][gen]["startup_capacity"] = p_min
            full_bids[t][gen]["shutdown_capacity"] = p_min

        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids


class FixedParametrizedBidder(ParametrizedBidder):

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
        storage_marginal_cost,
        storage_mw
    ):
        super().__init__(bidding_model_object,
                         day_ahead_horizon,
                         real_time_horizon,
                         n_scenario,
                         solver,
                         forecaster)
        self.wind_marginal_cost = 0
        self.wind_mw = self.bidding_model_object._design_params['wind_mw']
        self.storage_marginal_cost = storage_marginal_cost
        self.storage_mw = storage_mw

    def compute_day_ahead_bids(self, date, hour=0):
        gen = self.generator
        forecast = self.forecaster.forecast_day_ahead_capacity_factor(date, hour, gen, self.day_ahead_horizon)

        full_bids = {}

        for t_idx in range(self.day_ahead_horizon):
            da_wind = forecast[t_idx] * self.wind_mw
            bids = [(0, 0), (da_wind, 0), (da_wind + self.storage_mw, self.storage_marginal_cost)]
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
            full_bids[t][gen]["p_max"] = da_wind + self.storage_mw
            full_bids[t][gen]["startup_capacity"] = 0
            full_bids[t][gen]["shutdown_capacity"] = 0

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
            bids = [(0, 0), (rt_wind, 0), (rt_wind + self.storage_mw, self.storage_marginal_cost)]

            t = t_idx + hour
            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = convert_marginal_costs_to_actual_costs(bids)
            full_bids[t][gen]["p_min"] = 0
            full_bids[t][gen]["p_max"] = rt_wind + self.storage_mw
            full_bids[t][gen]["startup_capacity"] = 0
            full_bids[t][gen]["shutdown_capacity"] = 0

        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids
