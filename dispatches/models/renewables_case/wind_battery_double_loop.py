from dispatches.models.renewables_case.wind_battery_LMP import (
    wind_battery_mp_block,
    wind_battery_variable_pairs,
    wind_battery_periodic_variable_pairs,
)
from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration import Bidder, SelfScheduler
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
import pyomo.environ as pyo
import pandas as pd
from collections import deque
from functools import partial
from dispatches.models.renewables_case.load_parameters import wind_speeds
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))


def create_multiperiod_wind_battery_model(n_time_points):

    # create the multiperiod model object
    mp_wind_battery = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_mp_block, verbose=False),
        linking_variable_func=wind_battery_variable_pairs,
        periodic_variable_func=wind_battery_periodic_variable_pairs,
    )

    # initialize the wind resoure
    wind_resource = {
        t: {
            "wind_resource_config": {
                "resource_probability_density": {0.0: ((wind_speeds[t], 180, 1),)}
            }
        }
        for t in range(n_time_points)
    }

    mp_wind_battery.build_multi_period_model(wind_resource)

    return mp_wind_battery


def transform_design_model_to_operation_model(
    mp_wind_battery,
    wind_capacity=200e3,
    battery_power_capacity=25e3,
    battery_energy_capacity=100e3,
):

    blks = mp_wind_battery.get_active_process_blocks()

    for t, b in enumerate(blks):
        b.fs.windpower.system_capacity.fix(wind_capacity)
        b.fs.battery.nameplate_power.fix(battery_power_capacity)
        b.fs.battery.nameplate_energy.fix(battery_energy_capacity)

        if t == 0:
            b.fs.battery.initial_state_of_charge.fix()

        # deactivate periodic boundary condition
        if t == len(blks) - 1:
            b.periodic_constraints[0].deactivate()

    return


def update_wind_capacity_factor(mp_wind_battery, new_capacity_factors):

    blks = mp_wind_battery.get_active_process_blocks()
    for idx, b in enumerate(blks):
        b.fs.windpower.capacity_factor[0] = new_capacity_factors[idx]

    return


class MultiPeriodWindBattery:
    def __init__(
        self,
        model_data,
        wind_capacity_factors=None,
        wind_pmax_mw=200,
        battery_pmax_mw=25,
        battery_energy_capacity_mwh=100,
    ):
        """
        Arguments:

        Returns:
            Float64: Value of power output in last time step
        """

        self.model_data = model_data

        if wind_capacity_factors is None:
            raise ValueError("Please provide wind capacity factors.")
        self._wind_capacity_factors = wind_capacity_factors

        self._wind_pmax_mw = wind_pmax_mw
        self._battery_pmax_mw = battery_pmax_mw
        self._battery_energy_capacity_mwh = battery_energy_capacity_mwh

        self.result_list = []

    def populate_model(self, b, horizon):
        """
        Create a wind-battery model using the `MultiPeriod` package.
        Arguments:
            blk: this is an empty block passed in from eithe a bidder or tracker
        Returns:
             None
        """
        blk = b
        if not blk.is_constructed():
            blk.construct()

        blk.windBattery = create_multiperiod_wind_battery_model(horizon)
        transform_design_model_to_operation_model(
            mp_wind_battery=blk.windBattery,
            wind_capacity=self._wind_pmax_mw * 1e3,
            battery_power_capacity=self._battery_pmax_mw * 1e3,
            battery_energy_capacity=self._battery_energy_capacity_mwh * 1e3,
        )
        blk.windBattery_model = blk.windBattery.pyomo_model

        # deactivate any objective functions
        for obj in blk.windBattery_model.component_objects(pyo.Objective):
            obj.deactivate()

        # initialize time index for this block
        b._time_idx = pyo.Param(initialize=0, mutable=True)

        new_capacity_factors = self._get_capacity_factors(b)
        update_wind_capacity_factor(blk.windBattery, new_capacity_factors)

        active_blks = blk.windBattery.get_active_process_blocks()

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = pyo.Set(initialize=range(horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = (b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0]) * 1e-3
            blk.tot_cost[t] = b.fs.windpower.op_total_cost

        return

    def update_model(self, b, realized_soc, realized_energy_throughput):

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

        new_init_soc = round(realized_soc[-1], 2)
        active_blks[0].fs.battery.initial_state_of_charge.fix(new_init_soc)

        new_init_energy_throuput = round(realized_energy_throughput[-1], 2)
        active_blks[0].fs.battery.initial_energy_throughput.fix(
            new_init_energy_throuput
        )

        # shift the time -> update capacity_factor
        time_advance = min(len(realized_soc), 24)
        b._time_idx = pyo.value(b._time_idx) + time_advance

        new_capacity_factors = self._get_capacity_factors(b)
        update_wind_capacity_factor(mp_wind_battery, new_capacity_factors)

        return

    def _get_capacity_factors(self, b):

        horizon_len = len(b.windBattery.get_active_process_blocks())
        ans = self._wind_capacity_factors[pyo.value(b._time_idx) : pyo.value(b._time_idx) + horizon_len]

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

        realized_soc = deque(
            [
                pyo.value(active_blks[t].fs.battery.state_of_charge[0])
                for t in range(last_implemented_time_step + 1)
            ]
        )

        realized_energy_throughput = deque(
            [
                pyo.value(active_blks[t].fs.battery.energy_throughput[0])
                for t in range(last_implemented_time_step + 1)
            ]
        )

        return {
            "realized_soc": realized_soc,
            "realized_energy_throughput": realized_energy_throughput,
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
        mp_wind_battery = blk.windBattery
        active_blks = mp_wind_battery.get_active_process_blocks()

        df_list = []
        for t, process_blk in enumerate(active_blks):

            result_dict = {}

            result_dict["Generator"] = self.model_data.gen_name
            result_dict["Date"] = date
            result_dict["Hour"] = hour

            # simulation inputs
            result_dict["Horizon [hr]"] = int(t)

            # model vars
            result_dict["Total Wind Generation [MW]"] = float(
                round(pyo.value(process_blk.fs.windpower.electricity[0]) * 1e-3, 2)
            )
            result_dict["Total Power Output [MW]"] = float(
                round(pyo.value(blk.P_T[t]), 2)
            )
            result_dict["Wind Power Output [MW]"] = float(
                round(pyo.value(process_blk.fs.splitter.grid_elec[0] * 1e-3), 2)
            )
            result_dict["Battery Power Output [MW]"] = float(
                round(pyo.value(process_blk.fs.battery.elec_out[0] * 1e-3), 2)
            )
            result_dict["Wind Power to Battery [MW]"] = float(
                round(pyo.value(process_blk.fs.battery.elec_in[0] * 1e-3), 2)
            )
            result_dict["State of Charge [MWh]"] = float(
                round(pyo.value(process_blk.fs.battery.state_of_charge[0] * 1e-3), 2)
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


class SimpleForecaster:
    def __init__(self, horizon, n_sample, bus=309):
        self.horizon = horizon
        self.n_sample = n_sample
        self.price_df = (
            pd.read_csv(
                os.path.join(this_file_path, "wind_bus_lmps.csv"), index_col="Bus ID"
            )
            .loc[bus]
            .set_index("Date")
        )

    def forecast(self, date, hour):

        date = str(date)
        if self.horizon % 24 == 0:
            repeat = self.horizon // 24
        else:
            repeat = self.horizon // 24 + 1
        return {
            i: list(self.price_df.loc[date, "LMP"]) * repeat
            for i in range(self.n_sample)
        }


if __name__ == "__main__":

    run_track = True
    run_bid = True

    solver = pyo.SolverFactory("gurobi")

    if run_track:
        mp_wind_battery = MultiPeriodWindBattery(
            model_data=model_data,
            wind_capacity_factors=gen_capacity_factor,
        )

        n_tracking_hour = 1
        tracking_horizon = 4

        # create a `Tracker` using`mp_wind_battery`
        tracker_object = Tracker(
            tracking_model_object=mp_wind_battery,
            tracking_horizon=tracking_horizon,
            n_tracking_hour=n_tracking_hour,
            solver=solver,
        )

        # example market dispatch signal for 4 hours
        market_dispatch = [0, 3.5, 15.0, 24.5]

        # find a solution that tracks the dispatch signal
        tracker_object.track_market_dispatch(
            market_dispatch=market_dispatch, date="2021-07-26", hour="17:00"
        )

        # The tracked power output
        for t in range(4):
            print(f"Hour {t},Power output {pyo.value(tracker_object.power_output[t])}")

    if run_bid:

        day_ahead_horizon = 48
        real_time_horizon = 4
        n_scenario = 1
        mp_wind_battery_bid = MultiPeriodWindBattery(
            model_data=model_data,
            wind_capacity_factors=gen_capacity_factor,
        )

        backcaster = Backcaster(historical_da_prices, historical_rt_prices)

        bidder_object = SelfScheduler(
            bidding_model_object=mp_wind_battery_bid,
            day_ahead_horizon=day_ahead_horizon,
            real_time_horizon=real_time_horizon,
            n_scenario=n_scenario,
            solver=solver,
            forecaster=backcaster,
        )

        date = "2020-01-02"
        bids = bidder_object.compute_day_ahead_bids(date=date, hour=0)
        print(bids)
