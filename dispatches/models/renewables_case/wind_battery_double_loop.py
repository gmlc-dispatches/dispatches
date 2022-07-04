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
from dispatches.models.renewables_case.wind_battery_LMP import (
    wind_battery_mp_block,
    wind_battery_variable_pairs,
    wind_battery_periodic_variable_pairs,
)
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
import pyomo.environ as pyo
import pandas as pd
from collections import deque
from functools import partial
from dispatches.models.renewables_case.load_parameters import wind_speeds


def create_multiperiod_wind_battery_model(n_time_points, wind_cfs, input_params):
    """This function creates a MultiPeriodModel for the wind battery model.

    Args:
        n_time_points (int): number of time period for the model.

    Returns:
        MultiPeriodModel: a MultiPeriodModel for the wind battery model.
    """

    # create the multiperiod model object
    mp_wind_battery = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_mp_block, input_params=input_params, verbose=False),
        linking_variable_func=wind_battery_variable_pairs,
        periodic_variable_func=wind_battery_periodic_variable_pairs,
    )

    wind_capacity_factors = {t:
                            {'wind_resource_config': {
                                'capacity_factor': 
                                    [wind_cfs[t]]}} for t in range(n_time_points)}

    mp_wind_battery.build_multi_period_model(wind_capacity_factors)

    return mp_wind_battery


def transform_design_model_to_operation_model(
    mp_wind_battery,
    wind_capacity=200e3,
    battery_power_capacity=25e3,
    battery_energy_capacity=100e3,
):
    """Transform the multiperiod wind battery design model to operation model.

    Args:
        mp_wind_battery (MultiPeriodModel): a created multiperiod wind battery object
        wind_capacity (float, optional): wind farm capapcity in KW. Defaults to 200e3.
        battery_power_capacity (float, optional): battery power output capacity in KW. Defaults to 25e3.
        battery_energy_capacity (float, optional): battery energy capacity in KW. Defaults to 100e3.
    """

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
    """Update the wind capacity factor in the model during rolling horizon.

    Args:
        mp_wind_battery (MultiPeriodModel): a created multiperiod wind battery object
        new_capacity_factors (list): a list of new wind capacity
    """

    blks = mp_wind_battery.get_active_process_blocks()
    for idx, b in enumerate(blks):
        b.fs.windpower.capacity_factor[0] = new_capacity_factors[idx]

    return


class MultiPeriodWindBattery:
    def __init__(
        self,
        model_data,
        wind_capacity_factors=None,
        wind_pmax_mw=200.0,
        battery_pmax_mw=25.0,
        battery_energy_capacity_mwh=100.0,
    ):
        """Initialize a multiperiod wind battery model object for double loop.

        Args:
            model_data (GeneratorModelData): a GeneratorModelData that holds the generators params.
            wind_capacity_factors (list, optional): a list of wind capacity. Defaults to None.
            wind_pmax_mw (float, optional): wind farm capapcity in MW. Defaults to 200.
            battery_pmax_mw (float, optional): battery power output capapcity in MW. Defaults to 25.
            battery_energy_capacity_mwh (float, optional): battery energy capapcity in MW. Defaults to 100.

        Raises:
            ValueError: if wind capacity factor is not provided, ValueError will be raised
        """

        self.model_data = model_data

        if wind_capacity_factors is None:
            raise ValueError("Please provide wind capacity factors.")
        self._wind_capacity_factors = wind_capacity_factors

        self._wind_pmax_mw = wind_pmax_mw
        self._battery_pmax_mw = battery_pmax_mw
        self._battery_energy_capacity_mwh = battery_energy_capacity_mwh

        # a list that holds all the result in pd DataFrame
        self.result_list = []

    def populate_model(self, b, horizon):
        """Create a wind-battery model using the `MultiPeriod` package.

        Args:
            b (block): this is an empty block passed in from eithe a bidder or tracker
            horizon (int): the number of time periods
        """
        blk = b
        if not blk.is_constructed():
            blk.construct()

        input_params = {
            'wind_mw': self._wind_pmax_mw,
            'batt_mw': self._battery_pmax_mw
        }
        blk.windBattery = create_multiperiod_wind_battery_model(horizon, wind_cfs=self._wind_capacity_factors[0:horizon], input_params=input_params)
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

        active_blks = blk.windBattery.get_active_process_blocks()

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = pyo.Set(initialize=range(horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        blk.wind_waste_penalty = pyo.Param(default=1e3, mutable=True)
        blk.wind_waste = pyo.Expression(blk.HOUR)
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = (b.fs.splitter.grid_elec[0] + b.fs.battery.elec_out[0]) * 1e-3
            blk.wind_waste[t] = (b.fs.windpower.system_capacity * b.fs.windpower.capacity_factor[0] - b.fs.windpower.electricity[0]) * 1e-3
            blk.tot_cost[t] = b.fs.windpower.op_total_cost + b.fs.battery.var_cost + blk.wind_waste_penalty * blk.wind_waste[t]

        return

    def update_model(self, b, realized_soc, realized_energy_throughput):
        """Update variables using future wind capapcity the realized state-of-charge and enrgy throughput profiles.

        Args:
            b (block): the block that needs to be updated
            realized_soc (list): list of realized state of charge
            realized_energy_throughput (list): list of realized enrgy throughput
        """

        blk = b
        mp_wind_battery = blk.windBattery
        active_blks = mp_wind_battery.get_active_process_blocks()

        new_init_soc = round(realized_soc[-1], 2)
        active_blks[0].fs.battery.initial_state_of_charge.fix(new_init_soc)

        new_init_energy_throughput = round(realized_energy_throughput[-1], 2)
        active_blks[0].fs.battery.initial_energy_throughput.fix(
            new_init_energy_throughput
        )

        # shift the time -> update capacity_factor
        time_advance = min(len(realized_soc), 24)
        b._time_idx = pyo.value(b._time_idx) + time_advance

        new_capacity_factors = self._get_capacity_factors(b)
        update_wind_capacity_factor(mp_wind_battery, new_capacity_factors)

        return

    def _get_capacity_factors(self, b):
        """Fetch the future capacity factor.

        Args:
            b (block): the block that needs to be updated

        Returns:
            list: the capcity factors for the immediate future
        """

        horizon_len = len(b.windBattery.get_active_process_blocks())
        ans = self._wind_capacity_factors[
            pyo.value(b._time_idx) : pyo.value(b._time_idx) + horizon_len
        ]

        return ans

    @staticmethod
    def get_last_delivered_power(b, last_implemented_time_step):
        """Get last delivered power.

        Args:
            b (block): a multiperiod block
            last_implemented_time_step (int):  time index for the last implemented time period

        Returns:
            float: last delivered power
        """

        blk = b
        return pyo.value(blk.P_T[last_implemented_time_step])

    @staticmethod
    def get_implemented_profile(b, last_implemented_time_step):
        """Get implemented profiles, i.e., realized state-of-charge, energy throughput.

        Args:
            b (block): a multiperiod block
            last_implemented_time_step (int):  time index for the last implemented time period

        Returns:
            dict: dictionalry of implemented profiles.
        """

        blk = b
        mp_wind_battery = blk.windBattery
        active_blks = mp_wind_battery.get_active_process_blocks()

        realized_soc = deque(
            pyo.value(active_blks[t].fs.battery.state_of_charge[0])
            for t in range(last_implemented_time_step + 1)
        )

        realized_energy_throughput = deque(
            pyo.value(active_blks[t].fs.battery.energy_throughput[0])
            for t in range(last_implemented_time_step + 1)
        )

        return {
            "realized_soc": realized_soc,
            "realized_energy_throughput": realized_energy_throughput,
        }

    def record_results(self, b, date=None, hour=None, **kwargs):
        """Record the operations stats for the model, i.e., generator name, data, hour, horizon,
        total wind generation, total power output, wind power output, battery power output, charging power,
        state of charge, total costs.

        Args:
            b (block): a multiperiod block
            date (str, optional): current simulation date. Defaults to None.
            hour (int, optional): current simulation hour. Defaults to None.
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
            result_dict["Wind Curtailment [MW]"] = float(
                round(pyo.value(blk.wind_waste[0]), 2)
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
        """Write the saved results to a csv file.

        Args:
            path (str): the path to write the results.
        """

        pd.concat(self.result_list).to_csv(path, index=False)

    @property
    def power_output(self):
        return "P_T"

    @property
    def total_cost(self):
        return ("tot_cost", 1)
