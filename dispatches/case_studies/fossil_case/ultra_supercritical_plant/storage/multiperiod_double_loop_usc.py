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

"""
This script describes a multiperiod class to build an object for the
integrated ultra-supercritical power plant and motlen-salt based
thermal energy storage model.
"""

__author__ = "Naresh Susarla"

import pyomo.environ as pyo
import pandas as pd
from collections import deque
import idaes.logger as idaeslog

# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# DISPATCHES imports
from dispatches.case_studies.fossil_case.ultra_supercritical_plant \
    .storage.multiperiod_integrated_storage_usc import (create_usc_model,
                                                        usc_unfix_dof,
                                                        usc_custom_init,
                                                        get_usc_link_variable_pairs)


def create_multiperiod_usc_model(n_time_points=4, pmin=None, pmax=None):
    """
    Create a multi-period usc_mp cycle object. This object contains a pyomo
    model with a block for each time instance.

    n_time_points: Number of time blocks to create
    """
    multiperiod_usc = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=create_usc_model,
        initialization_func=usc_custom_init,
        unfix_dof_func=usc_unfix_dof,
        linking_variable_func=get_usc_link_variable_pairs,
        use_stochastic_build=False,
        outlvl=idaeslog.INFO,
        )

    flowsheet_options={"pmin": pmin,
                        "pmax": pmax}

    # create the multiperiod object
    multiperiod_usc.build_multi_period_model(
        model_data_kwargs={t: flowsheet_options for t in range(n_time_points)},
        flowsheet_options=flowsheet_options,
    )

    return multiperiod_usc


class MultiPeriodUsc:
    def __init__(
        self, model_data=None
    ):
        """
        Arguments:
            horizon::Int64 - number of time points to use for associated multi-period model

        Returns:
            Float64: Value of power output in last time step
        """
        self.multiperiod_usc = None
        self.result_list = []
        self.result_listimp = []
        self.model_data = model_data

    def populate_model(self, blk, horizon):
        """
        Create a integrated ultra-supercritical power plant and molten salt
        thermal energy storage model using the `MultiPeriod` package.

        Arguments:
            blk: this is an empty block passed in from eithe a bidder or tracker

        Returns:
             None
        """
        tank_min = 76000  # in kg
        tank_max = 6739292  # in kg

        if not blk.is_constructed():
            blk.construct()

        multiperiod_usc = create_multiperiod_usc_model(
            n_time_points=horizon,
            pmin=self.model_data.p_min, pmax=self.model_data.p_max
        )
        blk.usc_mp = multiperiod_usc

        active_blks = multiperiod_usc.get_active_process_blocks()
        active_blks[0].fs.previous_salt_inventory_hot.fix(tank_min)
        active_blks[0].fs.previous_salt_inventory_cold.fix(tank_max-tank_min)
        active_blks[0].fs.previous_power.fix(380)

        # create expression that references underlying power variables
        blk.HOUR = pyo.Set(initialize=range(horizon))
        blk.P_T = pyo.Expression(blk.HOUR)
        blk.tot_cost = pyo.Expression(blk.HOUR)
        blk.hot_level = pyo.Expression(blk.HOUR)
        blk.storage_power = pyo.Expression(blk.HOUR)
        blk.plant_power = pyo.Expression(blk.HOUR)
        blk.plant_duty = pyo.Expression(blk.HOUR)
        blk.hxc_salt = pyo.Expression(blk.HOUR)
        blk.hxc_duty = pyo.Expression(blk.HOUR)
        blk.hxc_salt_Tin = pyo.Expression(blk.HOUR)
        blk.hxc_salt_Tout = pyo.Expression(blk.HOUR)
        blk.hxd_salt = pyo.Expression(blk.HOUR)
        blk.hxd_duty = pyo.Expression(blk.HOUR)
        blk.hxd_salt_Tin = pyo.Expression(blk.HOUR)
        blk.hxd_salt_Tout = pyo.Expression(blk.HOUR)
        blk.hxd_steam_Tout = pyo.Expression(blk.HOUR)
        blk.hxd_steam_vfrac = pyo.Expression(blk.HOUR)
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = b.fs.net_power
            blk.hot_level[t] = b.fs.salt_inventory_hot
            blk.storage_power[t] = ((-1e-6)
                                 * b.fs.es_turbine.work_mechanical[0])
            blk.plant_duty[t] = b.fs.plant_heat_duty[0]
            blk.tot_cost[t] = (
                     b.fs.operating_cost
                     + (b.fs.plant_fixed_operating_cost
                     + b.fs.plant_variable_operating_cost) / (365 * 24)
                )
            blk.plant_power[t] = b.fs.plant_power_out[0]
            blk.hxc_salt[t] = b.fs.hxc.tube_inlet.flow_mass[0]
            blk.hxc_duty[t] = b.fs.hxc.heat_duty[0]
            blk.hxc_salt_Tin[t] = b.fs.hxc.tube_inlet.temperature[0]
            blk.hxc_salt_Tout[t] = b.fs.hxc.tube_outlet.temperature[0]
            blk.hxd_salt[t] = b.fs.hxd.shell_inlet.flow_mass[0]
            blk.hxd_duty[t] = b.fs.hxd.heat_duty[0]
            blk.hxd_salt_Tin[t] = b.fs.hxd.shell_inlet.temperature[0]
            blk.hxd_salt_Tout[t] = b.fs.hxd.shell_outlet.temperature[0]
            blk.hxd_steam_Tout[t] = b.fs.hxd.cold_side.properties_out[0].temperature
            blk.hxd_steam_vfrac[t] = b.fs.hxd.cold_side.properties_out[0].vapor_frac


        self.multiperiod_usc = multiperiod_usc
        return

    @staticmethod
    def update_model(b, implemented_power_output, realized_soc):

        """
        Update `blk` variables using the actual implemented power output.

        Arguments:
            blk: the block that needs to be updated
            realized soc, i.e. the hot salt storage tank level:
            implemented_power_output:

         Returns:
             None
        """
        multiperiod_usc = b.usc_mp
        active_blks = multiperiod_usc.get_active_process_blocks()

        implemented_power = round(implemented_power_output[-1])
        realized_soc = round(realized_soc[-1])
        print("Implemented Power (MPC)", implemented_power)
        print("Realized SOC (MPC)", realized_soc)

        active_blks[0].fs.previous_power.fix(implemented_power)
        active_blks[0].fs.previous_salt_inventory_hot.fix(realized_soc)

        return

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
        # blk = b
        return pyo.value(b.P_T[last_implemented_time_step])

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
        # blk = b
        multiperiod_usc = b.usc_mp
        active_blks = multiperiod_usc.get_active_process_blocks()
        implemented_power_output = deque(
            [
                pyo.value(active_blks[t].fs.net_power)
                for t in range(last_implemented_time_step + 1)
            ]
        )
        realized_soc = deque(
            [
                pyo.value(active_blks[t].fs.salt_inventory_hot)
                for t in range(last_implemented_time_step + 1)
            ]
        )

        return {
            "implemented_power_output": implemented_power_output,
            "realized_soc": realized_soc,
        }

    def record_results(self, blk, date=None, hour=None, **kwargs):

        """
        Record the operations stats for the model.

        Arguments:
            blk:  pyomo block
            date: current simulation date
            hour: current simulation hour

        Returns:
            None

        """
        # blk = b
        df_list = []
        df_listimp = []
        for t in blk.HOUR:
            result_dict = {}
            result_implemented = {}

            result_dict["Date"] = date
            result_dict["Hour"] = hour


            # simulation inputs
            result_dict["Horizon [hr]"] = int(t)

            # model vars
            result_dict["Thermal Power Generated [MW]"] = float(
                round(pyo.value(blk.P_T[t]), 2)
            )
            result_dict["Total Cost [$]"] = float(
                round(pyo.value(blk.tot_cost[t]), 2)
                )
            result_dict["Hot Tank Level [MT]"] = float(
                round(pyo.value(blk.hot_level[t]), 2)
                )
            result_dict["Plant Heat Duty [MWth]"] = float(
                round(pyo.value(blk.plant_duty[t]), 2)
                )
            result_dict["Storage Power [MW]"] = float(
                round(pyo.value(blk.storage_power[t]), 2)
                )
            result_dict["Plant Power [MW]"] = float(
                round(pyo.value(blk.plant_power[t]), 2)
                )
            result_dict["HXC Duty"] = float(
                round(pyo.value(blk.hxc_duty[t] * 1e-6), 2)
                )
            result_dict["HXD Duty"] = float(
                round(pyo.value(blk.hxd_duty[t] * 1e-6), 2)
                )
            result_dict["HXC Salt Flow"] = float(
                round(pyo.value(blk.hxc_salt[t]), 2)
                )
            result_dict["HXD Salt Flow"] = float(
                round(pyo.value(blk.hxd_salt[t]), 2)
                )
            result_dict["HXC Salt T in"] = float(
                round(pyo.value(blk.hxc_salt_Tin[t]), 2)
                )
            result_dict["HXC Salt T out"] = float(
                round(pyo.value(blk.hxc_salt_Tout[t]), 2)
                )
            result_dict["HXD Salt T in"] = float(
                round(pyo.value(blk.hxd_salt_Tin[t]), 2)
                )
            result_dict["HXD Salt T out"] = float(
                round(pyo.value(blk.hxd_salt_Tout[t]), 2)
                )
            result_dict["HXD Steam T out"] = float(
                round(pyo.value(blk.hxd_steam_Tout[t]), 2)
                )
            result_dict["HXD Steam out Vfrac"] = float(
                round(pyo.value(blk.hxd_steam_vfrac[t]), 2)
                )

            if t == 0:
                # simulation inputs
                result_implemented["Date"] = date
                result_implemented["Hour"] = hour
                result_implemented["Horizon [hr]"] = int(t)

                # model vars
                result_implemented["Thermal Power Generated [MW]"] = float(
                    round(pyo.value(blk.P_T[t]), 2)
                )
                result_implemented["Total Cost [$]"] = float(
                    round(pyo.value(blk.tot_cost[t]), 2)
                    )
                result_implemented["Hot Tank Level [MT]"] = float(
                    round(pyo.value(blk.hot_level[t]), 2)
                    )
                result_implemented["Plant Heat Duty [MWth]"] = float(
                    round(pyo.value(blk.plant_duty[t]), 2)
                    )
                result_implemented["Storage Power [MW]"] = float(
                    round(pyo.value(blk.storage_power[t]), 2)
                    )
                result_implemented["Plant Power [MW]"] = float(
                    round(pyo.value(blk.plant_power[t]), 2)
                    )
                result_implemented["HXC Duty"] = float(
                    round(pyo.value(blk.hxc_duty[t] * 1e-6), 2)
                    )
                result_implemented["HXD Duty"] = float(
                    round(pyo.value(blk.hxd_duty[t] * 1e-6), 2)
                    )
                result_implemented["HXC Salt Flow"] = float(
                    round(pyo.value(blk.hxc_salt[t]), 2)
                    )
                result_implemented["HXD Salt Flow"] = float(
                    round(pyo.value(blk.hxd_salt[t]), 2)
                    )
                result_implemented["HXC Salt T in"] = float(
                    round(pyo.value(blk.hxc_salt_Tin[t]), 2)
                    )
                result_implemented["HXC Salt T out"] = float(
                    round(pyo.value(blk.hxc_salt_Tout[t]), 2)
                    )
                result_implemented["HXD Salt T in"] = float(
                    round(pyo.value(blk.hxd_salt_Tin[t]), 2)
                    )
                result_implemented["HXD Salt T out"] = float(
                    round(pyo.value(blk.hxd_salt_Tout[t]), 2)
                    )
                result_implemented["HXD Steam T out"] = float(
                    round(pyo.value(blk.hxd_steam_Tout[t]), 2)
                    )
                result_implemented["HXD Steam out Vfrac"] = float(
                    round(pyo.value(blk.hxd_steam_vfrac[t]), 2)
                    )
            for key in kwargs:
                result_dict[key] = kwargs[key]
                result_implemented[key] = kwargs[key]

            result_df = pd.DataFrame.from_dict(result_dict, orient="index")
            df_list.append(result_df.T)
            result_df2 = pd.DataFrame.from_dict(result_implemented, orient="index")
            df_listimp.append(result_df2.T)

        # append to result list
        self.result_list.append(pd.concat(df_list))
        self.result_listimp.append(pd.concat(df_listimp))

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
        pd.concat(self.result_listimp).to_csv("tracking_results.csv")

    @property
    def power_output(self):
        return "P_T"

    @property
    def total_cost(self):
        return ("tot_cost", 1)
