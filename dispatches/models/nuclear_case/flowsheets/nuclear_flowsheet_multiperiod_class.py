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

__author__ = "Radhakrishna Tumbalam Gooty"

# General python imports
import numpy as np
import pandas as pd
from collections import deque

# Pyomo imports
from pyomo.environ import Set, Expression, value

# IDAES imports
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# Flowsheet function imports
from nuclear_flowsheet_fixed_design import build_ne_flowsheet


def get_nuclear_link_variable_pairs(t1, t2):
    """
    This function returns paris of variables that need to be connected across two time periods

    Args:
        t1: current time block
        t2: next time block

    Returns:
        None
    """
    return [
        (t1.fs.h2_tank.tank_holdup[0], t2.fs.h2_tank.tank_holdup_previous[0])
    ]


def create_multiperiod_nuclear_model(n_time_points=4):
    """
    This function creates a multi-period integrated nuclear flowsheet object. This object contains 
    a pyomo model with a block for each time instance.

    Args:
        n_time_points: Number of time blocks to create

    Returns:
        Object containing multi-period integrated nuclear flowsheet model
    """
    mp_nuclear = MultiPeriodModel(
        n_time_points,
        process_model_func=build_ne_flowsheet, 
        linking_variable_func=get_nuclear_link_variable_pairs
    )

    # create the multiperiod object
    mp_nuclear.build_multi_period_model()
    return mp_nuclear


class MultiPeriodNuclear:
    """
    This class builds an object containing multi-period integrated nuclear flowsheet model
    for the double loop workflow. 

    Args:
        horizon: Length of the time horizon
        pmin: Difference between power output from the nuclear power plant and the PEM capcity
        pmax: Power output from the nuclear power plant
        default_bid_curve: Default bid curve that needs to be used in the bidding process
        generator_name: Name of the generator in RTS - GMLC
    """
    def __init__(self, 
                 horizon, 
                 model_data):

        # If the default bid curve is not provided use the one below         
        # if default_bid_curve is None:
        #     self.default_bid_curve = {p: 15 for p in np.linspace(pmin, pmax, 5)}
        # else:
        #     self.default_bid_curve = default_bid_curve
        
        self.horizon = horizon
        self.mp_nuclear = None
        self.result_list = []
        self.model_data = model_data
        self.p_lower = model_data.p_min
        self.p_upper = model_data.p_max
        self.generator = model_data.gen_name

    def populate_model(self, blk):
        """
        This function creates the nuclear flowsheet model using the `MultiPeriod` class

        Args:
            blk: this is an empty block passed in from either a `Bidder` or `Tracker` object
        
        Returns:
             None
        """
        if not blk.is_constructed():
            blk.construct()

        mp_nuclear = create_multiperiod_nuclear_model(n_time_points=self.horizon)
        blk.nuclear = mp_nuclear
        blk.nuclear_model = mp_nuclear.pyomo_model

        active_blks = mp_nuclear.get_active_process_blocks()
        active_blks[0].fs.h2_tank.tank_holdup_previous.fix(0)

        # create expression that references underlying power variables in multi-period rankine
        blk.HOUR = Set(initialize=range(self.horizon))
        blk.P_T = Expression(blk.HOUR)
        blk.tot_cost = Expression(blk.HOUR)
        
        for (t, b) in enumerate(active_blks):
            blk.P_T[t] = b.fs.np_power_split.np_to_grid_port.electricity[0] * 1e-3
            blk.tot_cost[t] = b.fs.operating_cost

        self.mp_nuclear = mp_nuclear
        return

    @staticmethod
    def update_model(b, implemented_tank_holdup):

        """
        Update `b` variables using the actual implemented power output.

        Args:
            b: the block that needs to be updated
            implemented_tank_holdup: Updated the initial tank holdup

        Returns:
            None
        """
        mp_nuclear = b.nuclear
        active_blks = mp_nuclear.get_active_process_blocks()

        new_init_holdup = round(implemented_tank_holdup[-1])

        # update battery and power output based on implemented values
        active_blks[0].fs.h2_tank.tank_holdup_previous.fix(new_init_holdup)

        return

    @staticmethod
    def get_last_delivered_power(b, last_implemented_time_step):

        """
        Returns the last delivered power output.
        Args:
            b: the block
            last_implemented_time_step: time index for the last implemented time
                                        step
        Returns:
            Float64: Value of power output in last time step
        """
        return value(b.P_T[last_implemented_time_step])

    @staticmethod
    def get_implemented_profile(b, last_implemented_time_step):

        """
        This method gets the implemented variable profiles in the last optimization solve.

        Args:
            b: a Pyomo block
            last_implemented_time_step: time index for the last implemented time step

         Returns:
             profile: the intended profile, {unit: [...]}
        """
        active_blks = b.nuclear.get_active_process_blocks()
        implemented_tank_holdup = deque(
            [
                value(active_blks[t].fs.h2_tank.tank_holdup[0])
                for t in range(last_implemented_time_step + 1)
            ]
        )

        return {
            "implemented_tank_holdup": implemented_tank_holdup
        }

    def record_results(self, blk, date=None, hour=None, **kwargs):

        """
        Record the operations stats for the model.

        Args:
            blk:  pyomo block
            date: current simulation date
            hour: current simulation hour

        Returns:
            None
        """
        active_blks = blk.nuclear.get_active_process_blocks()
        df_list = []
        for t in blk.HOUR:
            result_dict = {
                "Date": date,
                "Hour": hour,
                "Horizon [hr]": int(t),
                "Power to Grid [MW]": float(round(value(blk.P_T[t]), 2)),
                "Power to PEM [MW]": float(round(value(active_blks[t].fs.pem.electricity[0]) * 1e-3, 2)),
                "Initial holdup [kg]": float(round(value(active_blks[t].fs.h2_tank.tank_holdup_previous[0])
                                                   * 2.016 * 1e-3, 2)),
                "Final holdup [kg]": float(round(value(active_blks[t].fs.h2_tank.tank_holdup[0])
                                                 * 2.016 * 1e-3, 2)),
                "Hydrogen Market [kg/hr]": float(round(value(active_blks[t].fs.h2_tank.outlet_to_pipeline.flow_mol[0])
                                                       * 2.016 * 1e-3 * 3600, 2)),
                "Total Cost [$]": float(round(value(blk.tot_cost[t]), 2))}

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
        
        Args:
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
    def pmin(self):
        # return self.generator_data['PMin MW']
        return self.p_lower
