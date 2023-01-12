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
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP import (
    wind_battery_pem_mp_block,
    wind_battery_pem_variable_pairs,
    wind_battery_pem_periodic_variable_pairs,
    h2_mols_per_kg, pem_bar, pem_temp
)
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
import pyomo.environ as pyo
import pandas as pd
from collections import deque
from functools import partial


def create_multiperiod_wind_pem_model(n_time_points, wind_cfs, input_params):
    """This function creates a MultiPeriodModel for the wind PEM model.

    Args:
        n_time_points (int): number of time period for the model.
        wind_cfs (sequence of floats): capacity factors of wind plant
        input_params (dict): system and cost parameters required for the Wind + PEM flowsheet

    Returns:
        MultiPeriodModel: a MultiPeriodModel for the wind PEM model.
    """
    if input_params['batt_mw'] != 0:
        raise ValueError("This model is not used for hybrids with battery. Battery MW must be 0.")

    # create the multiperiod model object
    mp_wind_pem = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_pem_mp_block, input_params=input_params, verbose=False),
        linking_variable_func=wind_battery_pem_variable_pairs,
        periodic_variable_func=wind_battery_pem_periodic_variable_pairs,
    )

    wind_capacity_factors = {t:
                            {'wind_resource_config': {
                                'capacity_factor': 
                                    [wind_cfs[t]]}} for t in range(n_time_points)}

    mp_wind_pem.build_multi_period_model(wind_capacity_factors)

    return mp_wind_pem


def transform_design_model_to_operation_model(
    mp_wind_pem,
    wind_capacity=200e3,
    pem_power_capacity=25e3,
):
    """Transform the multiperiod wind PEM design model to operation model.

    Fix the sizes and deactivate periodic boundary condition

    Args:
        mp_wind_pem (MultiPeriodModel): a created multiperiod wind PEM object
        wind_capacity (float, optional): wind farm capapcity in KW. Defaults to 200e3.
        PEM (float, optional): PEM power output capacity in KW. Defaults to 25e3.
    """

    m = mp_wind_pem.pyomo_model
    blks = mp_wind_pem.get_active_process_blocks()
    m.pem_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=pem_power_capacity, units=pyo.units.kW)

    for t, b in enumerate(blks):
        b.fs.windpower.system_capacity.fix(wind_capacity)
        b.fs.pem.max_p = pyo.Constraint(expr=b.fs.pem.electricity[0] <= m.pem_system_capacity)
        # deactivate periodic boundary condition
        if t == len(blks) - 1:
            b.periodic_constraints[0].deactivate()

    return


def update_wind_capacity_factor(mp_wind_pem, new_capacity_factors):
    """Update the wind capacity factor in the model during rolling horizon.

    Args:
        mp_wind_pem (MultiPeriodModel): a created multiperiod wind PEM object
        new_capacity_factors (list): a list of new wind capacity
    """

    blks = mp_wind_pem.get_active_process_blocks()
    for idx, b in enumerate(blks):
        b.fs.windpower.capacity_factor[0] = new_capacity_factors[idx]

    return


class MultiPeriodWindPEM:
    def __init__(
        self,
        model_data,
        wind_capacity_factors,
        wind_pmax_mw=200.0,
        pem_pmax_mw=25.0,
    ):
        """Initialize a multiperiod wind PEM model object for double loop.

        Args:
            model_data (GeneratorModelData): a GeneratorModelData that holds the generators params.
            wind_capacity_factors (list): a list of wind capacity.
            wind_pmax_mw (float, optional): wind farm capapcity in MW. Defaults to 200.
            pem_pmax_mw (float, optional): PEM power output capapcity in MW. Defaults to 25.

        Raises:
            ValueError: if wind capacity factor is not provided, ValueError will be raised
        """

        self.model_data = model_data

        if wind_capacity_factors is None:
            raise ValueError("Please provide wind capacity factors.")
        self._wind_capacity_factors = wind_capacity_factors

        self._wind_pmax_mw = wind_pmax_mw
        self._pem_pmax_mw = pem_pmax_mw

        # a list that holds all the result in pd DataFrame
        self.result_list = []

    def populate_model(self, b, horizon):
        """Create a wind-PEM model using the `MultiPeriod` package.

        The flowsheet model has option for battery; makes sure the battery size is 0 for Wind + PEM

        Args:
            b (block): this is an empty block passed in from eithe a bidder or tracker
            horizon (int): the number of time periods
        """
        blk = b
        if not blk.is_constructed():
            blk.construct()

        input_params = {
            'wind_mw': self._wind_pmax_mw,
            'pem_mw': self._pem_pmax_mw,
            "pem_bar": pem_bar,
            "pem_temp": pem_temp,
            'batt_mw': 0
        }
        blk.windPEM = create_multiperiod_wind_pem_model(horizon, wind_cfs=self._wind_capacity_factors[0:horizon], input_params=input_params)
        transform_design_model_to_operation_model(
            mp_wind_pem=blk.windPEM,
            wind_capacity=self._wind_pmax_mw * 1e3,
            pem_power_capacity=self._pem_pmax_mw * 1e3,
        )

        # deactivate any objective functions
        for obj in blk.windPEM.component_objects(pyo.Objective):
            obj.deactivate()

        # initialize time index for this block
        b._time_idx = pyo.Param(initialize=0, mutable=True)

        active_blks = blk.windPEM.get_active_process_blocks()

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = pyo.Set(initialize=range(horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        blk.wind_waste = pyo.Expression(blk.HOUR)

        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = b.fs.splitter.grid_elec[0] * 1e-3
            blk.wind_waste[t] = (b.fs.windpower.system_capacity * b.fs.windpower.capacity_factor[0] - b.fs.windpower.electricity[0]) * 1e-3
            blk.tot_cost[t] = (b.fs.windpower.system_capacity * b.fs.windpower.op_cost / 8760 
                             + blk.windPEM.pem_system_capacity * b.fs.pem.op_cost / 8760 + b.fs.pem.var_cost)
        return

    def update_model(self, b, realized_h2_sales):
        """Update variables using future wind capacity and the realized hydrogen sales.

        The hydrogen sales aren't stored at the moment.

        Args:
            b (block): the block that needs to be updated
        """

        blk = b
        mp_wind_pem = blk.windPEM

        # shift the time -> update capacity_factor
        time_advance = min(len(realized_h2_sales), 24)
        b._time_idx = pyo.value(b._time_idx) + time_advance

        new_capacity_factors = self._get_capacity_factors(b)
        update_wind_capacity_factor(mp_wind_pem, new_capacity_factors)

        return

    def _get_capacity_factors(self, b):
        """Fetch the future capacity factor.

        Args:
            b (block): the block that needs to be updated

        Returns:
            list: the capcity factors for the immediate future
        """

        horizon_len = len(b.windPEM.get_active_process_blocks())
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
        """Get implemented profiles, i.e., realized hydrogen sales.

        Args:
            b (block): a multiperiod block
            last_implemented_time_step (int):  time index for the last implemented time period

        Returns:
            dict: dictionalry of implemented profiles.
        """

        blk = b
        mp_wind_pem = blk.windPEM
        active_blks = mp_wind_pem.get_active_process_blocks()

        realized_h2_sales = deque(
            pyo.value(active_blks[t].fs.pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600)
            for t in range(last_implemented_time_step + 1)
        )

        return {
            "realized_h2_sales": realized_h2_sales,
        }

    def record_results(self, b, date=None, hour=None, **kwargs):
        """Record the operations stats for the model, i.e., generator name, data, hour, horizon,
        total wind generation, total power output, wind power output, wind to PEM output, hydrogen sales, total costs.

        Args:
            b (block): a multiperiod block
            date (str, optional): current simulation date. Defaults to None.
            hour (int, optional): current simulation hour. Defaults to None.
        """
        blk = b
        mp_wind_pem = blk.windPEM
        active_blks = mp_wind_pem.get_active_process_blocks()

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
            result_dict["Wind to PEM [MW]"] = float(
                round(pyo.value(process_blk.fs.splitter.pem_elec[0] * 1e-3), 2)
            )
            result_dict["Wind Curtailment [MW]"] = float(
                round(pyo.value(blk.wind_waste[0]), 2)
            )
            result_dict["Hydrogen Sales [kg]"] = float(
                round(pyo.value(process_blk.fs.pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600), 2)
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
