from idaes.apps.grid_integration.bidder import *
from idaes.apps.grid_integration.forecaster import AbstractPrescientPriceForecaster

class FileForecaster(AbstractPrescientPriceForecaster):

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data["DateTime"] = self.data['Unnamed: 0']
        self.data.drop('Unnamed: 0', inplace=True, axis=1)
        self.data.index = pd.to_datetime(self.data["DateTime"])

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
                         forecaster)
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

            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = [(power, 0), (max_power, self.battery_marginal_cost)]
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
                full_bids[t][gen]["p_cost"] = [(realized_day_ahead_dispatches[t % 24], 0), (power, rt_price[t_idx])]
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
