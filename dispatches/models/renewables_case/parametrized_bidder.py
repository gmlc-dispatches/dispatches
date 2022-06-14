from idaes.apps.grid_integration.bidder import *

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

        """
        Initializes the stochastic bidder object.

        Arguments:
            bidding_model_object: the model object for bidding

            day_ahead_horizon: number of time periods in the day-ahead bidding problem

            real_time_horizon: number of time periods in the real-time bidding problem

            n_scenario: number of uncertain LMP scenarios

            solver: a Pyomo mathematical programming solver object

            forecaster: an initialized LMP forecaster object

        Returns:
            None
        """

        self.bidding_model_object = bidding_model_object
        self.day_ahead_horizon = day_ahead_horizon
        self.real_time_horizon = real_time_horizon
        self.n_scenario = n_scenario
        self.solver = solver
        self.forecaster = forecaster

        self._check_inputs()

        self.generator = self.bidding_model_object.model_data.gen_name

        # day-ahead model
        self.day_ahead_model = self.formulate_DA_bidding_problem()
        self.real_time_model = self.formulate_RT_bidding_problem()

        # declare a list to store results
        self.bids_result_list = []

    def formulate_DA_bidding_problem(self):
        pass

    def formulate_RT_bidding_problem(self):
        pass

    def compute_day_ahead_bids(self, date, hour=0):
        gen = self.generator
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.forecaster[self.forecaster.index >= datetime_index].head(self.day_ahead_horizon)
        da_wind = forecast[f'{gen}-DACF'].values * self.bidding_model_object._wind_pmax_mw

        gen = self.generator

        full_bids = {}

        for t_idx in range(self.day_ahead_horizon):
            power = da_wind[t_idx] * 0.8
            max_power = da_wind[t_idx]
            t = t_idx + hour

            full_bids[t] = {}
            full_bids[t][gen] = {}
            full_bids[t][gen]["p_cost"] = [(power, 0), (max_power, 30)]
            full_bids[t][gen]["p_min"] = max_power
            full_bids[t][gen]["p_max"] = max_power
            full_bids[t][gen]["startup_capacity"] = max_power
            full_bids[t][gen]["shutdown_capacity"] = max_power

        self._record_bids(full_bids, date, hour, Market="Day-ahead")
        return full_bids

    def compute_real_time_bids(
        self, date, hour, realized_day_ahead_prices, realized_day_ahead_dispatches, battery_starting_energy
    ):
        gen = self.generator
        datetime_index = pd.to_datetime(date) + pd.Timedelta(hours=hour)
        forecast = self.forecaster[self.forecaster.index >= datetime_index].head(self.real_time_horizon)
        remaining_wind_energy = forecast[f'{gen}-RTCF'].values * self.bidding_model_object._wind_pmax_mw # - realized_day_ahead_dispatches[hour:hour + self.real_time_horizon]

        rt_price = [i / 0.9 for i in realized_day_ahead_prices]
        gen = self.generator

        full_bids = {}

        for t_idx in range(self.real_time_horizon):
            # make sure bids are convex when DA CF > RT CF
            # power = remaining_wind_energy[t_idx] + battery_starting_energy / self.real_time_horizon * 1e-3
            power = remaining_wind_energy[t_idx] * 0.8
            max_power = remaining_wind_energy[t_idx]

            t = t_idx + hour

            full_bids[t] = {}
            full_bids[t][gen] = {}
            # if power > realized_day_ahead_dispatches[t]:
            #     full_bids[t][gen]["p_cost"] = [(realized_day_ahead_dispatches[t], 0), (power, rt_price[t_idx])]
            # else:
            #     full_bids[t][gen]["p_cost"] = [(power, 0)]
            full_bids[t][gen]["p_cost"] = [(power, 0), (max_power, 30)]
            full_bids[t][gen]["p_min"] = max_power
            full_bids[t][gen]["p_max"] = max_power
            full_bids[t][gen]["startup_capacity"] = max_power
            full_bids[t][gen]["shutdown_capacity"] = max_power

        self._record_bids(full_bids, date, hour, Market="Real-time")
        return full_bids

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

    def update_real_time_model(self, **kwargs):
        pass

    def update_day_ahead_model(self, **kwargs):
        pass