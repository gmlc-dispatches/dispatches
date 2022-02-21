from wind_battery_LMP import wind_battery_optimize
from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration import Bidder, SelfScheduler
from idaes.apps.grid_integration import DoubleLoopCoordinator
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from collections import deque


def transform_design_model_to_operation_model(mp_wind_battery):

    blks = mp_wind_battery.get_active_process_blocks()

    for b in blks:
        b.fs.windpower.system_capacity.fix()
        b.fs.battery.nameplate_power.fix()

        ## TODO: deactivate periodic boundary condition??

    return


def update_wind_capacity_factor(mp_wind_battery, new_capacity_factors):

    blks = mp_wind_battery.get_active_process_blocks()
    for idx, b in enumerate(blks):
        b.fs.windpower.capacity_factor[0] = new_capacity_factors[idx]

    return


wind_generator = "309_WIND_1"
capacity_factor_df = pd.read_csv("capacity_factors.csv")
gen_capacity_factor = list(capacity_factor_df[wind_generator])


class MultiPeriodWindBattery:
    def __init__(
        self,
        horizon=4,
        pmin=0,
        pmax=200,
        wind_capacity_factors=None,
        default_bid_curve=None,
        generator_name=wind_generator,
    ):
        """
        Arguments:
            horizon::Int64 - number of time points to use for associated multi-period model
        Returns:
            Float64: Value of power output in last time step
        """
        if default_bid_curve is None:
            self.default_bid_curve = {p: 30 for p in np.linspace(pmin, pmax, 5)}
        else:
            self.default_bid_curve = default_bid_curve

        if wind_capacity_factors is None:
            raise ValueError("Please provide wind capacity factors.")
        self._wind_capacity_factors = wind_capacity_factors

        self.horizon = horizon
        self.mp_rankine = None
        self.result_list = []
        self.p_lower = pmin
        self.p_upper = pmax
        self.generator = generator_name

        self._time_idx = 0

    def populate_model(self, b):
        """
        Create a rankine-cycle-battery model using the `MultiPeriod` package.
        Arguments:
            blk: this is an empty block passed in from eithe a bidder or tracker
        Returns:
             None
        """
        blk = b
        if not blk.is_constructed():
            blk.construct()

        blk.windBattery = wind_battery_optimize()
        transform_design_model_to_operation_model(blk.windBattery)
        blk.windBattery_model = blk.windBattery.pyomo_model
        blk.windBattery_model.obj.deactivate()

        new_capacity_factors = self._get_capacity_factors()
        update_wind_capacity_factor(blk.windBattery, new_capacity_factors)

        active_blks = blk.windBattery.get_active_process_blocks()

        # TODO
        # active_blks[0].battery.previous_soc.fix(0)
        # active_blks[0].rankine.previous_power_output.fix(50.0)

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = pyo.Set(initialize=range(self.horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = (b.fs.wind_to_grid[0] + b.fs.battery.elec_out[0]) * 1e-3
            blk.tot_cost[t] = b.fs.windpower.op_total_cost

        return

    def update_model(self, b, realized_soc):

        """
        Update `blk` variables using the actual implemented power output.
        Arguments:
            blk: the block that needs to be updated
         Returns:
             None
        """
        blk = b
        mp_wind_battery = blk.windBattery
        active_blks = mp_wind_battery.get_active_process_blocks()

        # implemented_power = round(implemented_power_output[-1])
        realized_soc = round(realized_soc[-1])

        # update battery and power output based on implemented values
        # active_blks[0].rankine.previous_power_output.fix(implemented_power)
        active_blks[0].fs.battery.initial_state_of_charge.fix(realized_soc)

        new_capacity_factors = self._get_capacity_factors()
        update_wind_capacity_factor(mp_wind_battery, new_capacity_factors)

        return

    def _get_capacity_factors(self):
        ans = self._wind_capacity_factors[
            self._time_idx : self._time_idx + self.horizon
        ]
        self._time_idx += self.horizon
        return ans

    @staticmethod
    def get_last_delivered_power(b, last_implemented_time_step):

        """
        Returns the last delivered power output.
        Arguments:
            blk: the block
            last_implemented_time_step: time index for the last implemented time
                                        step
        Returns:
            Float64: Value of power output in last time step
        """
        blk = b
        return pyo.value(blk.P_T[last_implemented_time_step])

    @staticmethod
    def get_implemented_profile(b, last_implemented_time_step):

        """
        This method gets the implemented variable profiles in the last optimization solve.
        Arguments:
            blk: a Pyomo block
            last_implemented_time_step: time index for the last implemented time step
         Returns:
             profile: the intended profile, {unit: [...]}
        """
        blk = b
        mp_wind_battery = blk.windBattery
        active_blks = mp_wind_battery.get_active_process_blocks()
        # implemented_wind_power_output = deque(
        #    [
        #        pyo.value(active_blks[t].fs.wind_to_grid[0])
        #        for t in range(last_implemented_time_step + 1)
        #    ]
        # )
        realized_soc = deque(
            [
                pyo.value(active_blks[t].fs.battery.state_of_charge[0])
                for t in range(last_implemented_time_step + 1)
            ]
        )

        return {
            # "implemented_wind_power_output": implemented_wind_power_output,
            "realized_soc": realized_soc,
        }

    def record_results(self, b, date=None, hour=None, **kwargs):

        """
        Record the operations stats for the model.
        Arguments:
            blk:  pyomo block
            date: current simulation date
            hour: current simulation hour
        Returns:
            None
        """
        blk = b
        df_list = []
        for t in blk.HOUR:
            result_dict = {}

            # result_dict['Generator'] = self.generator
            result_dict["Date"] = date
            result_dict["Hour"] = hour

            # simulation inputs
            result_dict["Horizon [hr]"] = int(t)

            # model vars
            result_dict["Total Power Output [MW]"] = float(
                round(pyo.value(blk.P_T[t]), 2)
            )
            result_dict["Total Cost [$]"] = float(round(pyo.value(blk.tot_cost[t]), 2))

            for key in kwargs:
                result_dict[key] = kwargs[key]

            result_df = pd.DataFrame.from_dict(result_dict, orient="index")
            df_list.append(result_df.T)

        # append to result list
        self.result_list.append(pd.concat(df_list))

        return

    def write_results(self, path):

        """
        Write the saved results to a csv file.
        Arguments:
            path: the path to write the results.
        Return:
            None
        """

        pd.concat(self.result_list).to_csv(path, index=False)

    @property
    def power_output(self):
        return "P_T"

    @property
    def total_cost(self):
        return ("tot_cost", 1)

    @property
    def default_bids(self):
        return self.default_bid_curve

    @property
    def pmin(self):
        return self.p_lower


class SimpleForecaster:
    def __init__(self, horizon, n_sample):
        self.horizon = horizon
        self.n_sample = n_sample

    def forecast(self, date, hour, prediction):
        return {i: [prediction] * self.horizon for i in range(self.n_sample)}


if __name__ == "__main__":

    run_track = False
    run_bid = True

    if run_track:
        mp_wind_battery = MultiPeriodWindBattery(
            wind_capacity_factors=gen_capacity_factor
        )

        n_tracking_hour = (
            1  # frequency we perform tracking (e.g. 1 mean at each hourly interval)
        )
        solver = pyo.SolverFactory("ipopt")

        # create a `Tracker` using`mp_wind_battery`
        tracker_object = Tracker(
            tracking_model_object=mp_wind_battery,
            n_tracking_hour=n_tracking_hour,
            solver=solver,
        )

        # example market dispatch signal for 4 hours
        market_dispatch = [50, 60, 55, 70]

        # find a solution that tracks the dispatch signal
        tracker_object.track_market_dispatch(
            market_dispatch=market_dispatch, date="2021-07-26", hour="17:00"
        )

        # The tracked power output
        for t in range(4):
            print(f"Hour {t},Power output {pyo.value(tracker_object.power_output[t])}")

    if run_bid:

        bidding_horizon = 24
        n_scenario = 1
        pmin = 0
        pmax = 200
        mp_rankine_bid = MultiPeriodWindBattery(
            horizon=bidding_horizon,
            pmin=pmin,
            pmax=pmax,
            wind_capacity_factors=gen_capacity_factor,
        )

        solver = pyo.SolverFactory("ipopt")

        bidder_object = SelfScheduler(
            bidding_model_object=mp_rankine_bid,
            n_scenario=n_scenario,
            horizon=bidding_horizon,
            solver=solver,
            forecaster=SimpleForecaster(horizon=bidding_horizon, n_sample=n_scenario),
        )

        # bidder_object = Bidder(
        #     bidding_model_object=mp_rankine_bid,
        #     n_scenario=n_scenario,
        #     solver=solver,
        #     forecaster=SimpleForecaster(horizon=bidding_horizon, n_sample=n_scenario),
        # )

        date = "2021-08-20"
        bids = bidder_object.compute_bids(date=date, hour=None, prediction=31.0)
        print(bids)
