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
from idaes.apps.grid_integration.bidder import *
from idaes.apps.grid_integration.forecaster import AbstractPrescientPriceForecaster


class PerfectForecaster(AbstractPrescientPriceForecaster):
    def __init__(self, data_path_or_df):
        """
        Perfect forecaster that reads the data from a Dataframe containing:
         - {bus}-DALMP
         - {bus}-RTLMP
         - {bus}-DACF and {bus}-RTCF for renewable plants
        """
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

    def get_column_from_data(self, date, hour, horizon, col):
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.data[self.data.index >= datetime_index].head(horizon)
        values = forecast[col].values
        if len(values) < horizon:
            values = np.append(values, self.data[col].values[:horizon - len(values)])
        return values

    def forecast_day_ahead_prices(self, date, hour, bus, horizon, _):
        return self.get_column_from_data(date, hour, horizon, f'{bus}-DALMP')

    def forecast_real_time_prices(self, date, hour, bus, horizon, _):
        return self.get_column_from_data(date, hour, horizon, f'{bus}-RTLMP')

    def forecast_day_ahead_capacity_factor(self, date, hour, gen, horizon):
        return self.get_column_from_data(date, hour, horizon, f'{gen}-DACF')

    def forecast_real_time_capacity_factor(self, date, hour, gen, horizon):
        return self.get_column_from_data(date, hour, horizon, f'{gen}-RTCF')


class ParametrizedBidder(AbstractBidder):

    """
    Template class for bidders that uses parameters to create the DA and RT bid curves
    """
    def __init__(
        self,
        bidding_model_object,
        day_ahead_horizon,
        real_time_horizon,
        solver,
        forecaster,
    ):
        self.bidding_model_object = bidding_model_object
        self.day_ahead_horizon = day_ahead_horizon
        self.real_time_horizon = real_time_horizon
        self.n_scenario = 1
        self.solver = solver
        self.forecaster = forecaster
        self.real_time_underbid_penalty = 500

        self._check_inputs()

        self.generator = self.bidding_model_object.model_data.gen_name

        self.bids_result_list = []

        self.battery_marginal_cost = 25
        self.battery_capacity_ratio = 0.4

    def formulate_DA_bidding_problem(self):
        pass

    def formulate_RT_bidding_problem(self):
        pass

    def compute_day_ahead_bids(self, date, hour=0):
        raise NotImplementedError

    def compute_real_time_bids(
        self, date, hour, realized_day_ahead_prices, realized_day_ahead_dispatches, tracker_profile
    ):
        raise NotImplementedError

    def update_real_time_model(self, **kwargs):
        pass

    def update_day_ahead_model(self, **kwargs):
        pass

    def record_bids(self, bids, model, date, hour, market):

        """
        This function records the bids and the details in the underlying bidding model.

        Arguments:
            bids: the obtained bids for this date.

            model: bidding model

            date: the date we bid into

            hour: the hour we bid into

        Returns:
            None

        """

        # record bids
        self._record_bids(bids, date, hour, Market=market)

        # record the details of bidding model
        for i in model.SCENARIOS:
            self.bidding_model_object.record_results(
                model.fs[i], date=date, hour=hour, Scenario=i, Market=market
            )

        return

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

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, name):
        self._generator = name

