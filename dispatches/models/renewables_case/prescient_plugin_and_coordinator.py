## Market (Prescient) Interfacing Plugin and Coordinator for a Bidding Wind+Storage Hybrid System
# By Abinet and Bernard, NREL 

import os
import pandas as pd
import numpy as np
import math
import dateutil.parser
from pyomo.common.config import ConfigDict, ConfigValue

from wind_power_forecaster import WindPowerForecaster
from lmp_forecaster import LMPForecaster
from bidder import Bidder
from tracker import Tracker

from dispatches.models.renewables_case.wind_battery_LMP import wind_battery_optimize, record_results

name = 'Wind+Storage Hybrid System Example'

print(f"Loading plugin {name}")

def _print_msg(msg):
    print(f"FROM PLUGIN {name}: {msg}")

current_dir = os.getcwd()

## RTS-GMLC dir

rts_gmlc_dir = "/Users/dguittet/Projects/Dispatches/Prescient/downloads/rts_gmlc"
bidding_plugin_dir = "/Users/dguittet/Projects/Dispatches/workspace"

## Get the wind power plant and energy storage system capacities from the conceptual design 

mp_wind_battery = wind_battery_optimize()
(
    soc,
    wind_gen,
    batt_to_grid,
    wind_to_grid,
    wind_to_batt,
    elec_revenue,
    lmp,
    wind_cap,
    batt_cap,
    rev,
    npv,
) = record_results(mp_wind_battery)

_print_msg(f"Wind Power Plant Designed Capacity: {wind_cap}")
_print_msg(f"Energy Storage System Designed Capacity: {batt_cap}")

## Bidding Config

bidding_config = {"gen_name" : ['303_WIND_1'],
                  "gen_cap" : [wind_cap],                   
                  "storage_hour" : 4,
                  "storage_cap" : batt_cap*4,
                  "storage_charg_eff" : 0.95,
                  "storage_discharg_eff" : 0.90,
                  "storage_mileage_cost" : 0.0,
                  "storage_min_soc" : 0.2,
                  "storage_max_soc" : 1.0,
                  "storage_initial_soc" : 0.2,              
                  "bidding_date" : '2020-07-11',
                  "bidding_horizon" : 24
                  }

## Instantiate the wind and LMP forecasters 

wind_power_forecaster = WindPowerForecaster(rts_gmlc_dir)
lmp_forecaster = LMPForecaster(bidding_plugin_dir)

## Specify the bidding date(s)

bidding_date = bidding_config["bidding_date"]
bidding_date1 = '2020-07-11'
bidding_date2 = '2020-07-12'

## Get wind power forecasts and actuals

bus = '303_WIND_1'
date = bidding_date
horizon = bidding_config["bidding_horizon"]

#wind_power_forecast, wind_power_actual = wind_power_forecaster.process_forecast(bus, date, horizon)

wind_power_forecast = wind_power_forecaster.process_forecast(bus, bidding_date1, horizon)[0] + \
    wind_power_forecaster.process_forecast(bus, bidding_date2, horizon)[0]

wind_power_actual = wind_power_forecaster.process_forecast(bus, bidding_date1, horizon)[1] + \
    wind_power_forecaster.process_forecast(bus, bidding_date2, horizon)[1]

## Get DA and RT LMPs

bus_name = 'Caesar'                            

#DA_LMP, RT_LMP = lmp_forecaster.process_LMPs(bus_name, date, horizon)

DA_LMP = lmp_forecaster.process_LMPs(bus_name, bidding_date1, horizon)[0] + \
    lmp_forecaster.process_LMPs(bus_name, bidding_date2, horizon)[0]

RT_LMP = lmp_forecaster.process_LMPs(bus_name, bidding_date1, horizon)[1] + \
    lmp_forecaster.process_LMPs(bus_name, bidding_date2, horizon)[1]

## Instantiate the bidder

solver = 'xpress_direct' # 'cbc' or other solvers can be used

bidder = Bidder(bidding_config, wind_power_forecast, wind_power_actual, DA_LMP, RT_LMP, solver)

## Instantiate the tracker

tracking_config = {"gen_name" : ['303_WIND_1'],                   
                   "tracking_date" : bidding_date
                  }

tracker = Tracker(tracking_config, solver)

bidding_and_tracking_output_dir = current_dir + '/bidding_and_tracking_outputs' 
if not os.path.exists(bidding_and_tracking_output_dir):
    os.mkdir(bidding_and_tracking_output_dir)

## Specify the plugin specifications/options

def get_configuration(key):

    config = ConfigDict()

    config.declare('track_ruc_signal', 
                   ConfigValue(domain=bool,
                               description='When tracking the market signal, RUC signals are used instead of the SCED signal.',
                               default=True)).declare_as_argument(f'--track-ruc-signal')

    config.declare('track_sced_signal', 
                   ConfigValue(domain=bool,
                               description='When tracking the market signal, SCED signals are used instead of the RUC signal.',
                               default=True)).declare_as_argument(f'--track-sced-signal')
            
    config.declare('track_horizon', 
                   ConfigValue(domain=int,
                               description='Specifies the number of hours in the look-ahead horizon when each tracking process is executed.',
                               default=48)).declare_as_argument(f'--track-horizon')

    config.declare('bidding_generator', 
                   ConfigValue(domain=list,
                               description='Specifies the generator we derive bidding strategis for.',
                               default=['303_WIND_1'])).declare_as_argument(f'--bidding-generator')     # ['122_WIND_1', '303_WIND_1', '309_WIND_1', '317_WIND_1']
    
    config.declare('bidding', 
                   ConfigValue(domain=bool,
                               description='Invoke generator strategic bidding strategy when simulate.',
                               default=True)).declare_as_argument(f'--bidding')
        
    return config


## Register the plugins

def register_plugins(context,
                     options,
                     plugin_options):

    def initialize_plugin(options, simulator):

        _print_msg(f"In initialize_plugin in {name}")                     

        _print_msg(f"bidding_generator: {plugin_options.bidding_generator}")

        _print_msg(f"ruc_horizon: {options.ruc_horizon}")

        ## Dicts to store results

            # 1. RUC and SCED dispatch levels
        
        simulator.data_manager.extensions['ruc_schedule'] = dict()
        simulator.data_manager.extensions['sced_schedule'] = dict()

            # 2. DA quantities
                     
        simulator.data_manager.extensions['wind_produced_DA'] = dict()      
        simulator.data_manager.extensions['wind_to_grid_DA'] = dict() 
        simulator.data_manager.extensions['wind_to_storage_DA'] = dict()           
        simulator.data_manager.extensions['storage_to_grid_DA'] = dict()  
        simulator.data_manager.extensions['hybrid_sys_bid_DA'] = dict() 

            # 3. RT quantities    
                     
        simulator.data_manager.extensions['wind_produced_RT'] = dict()      
        simulator.data_manager.extensions['wind_to_grid_RT'] = dict() 
        simulator.data_manager.extensions['wind_to_storage_RT'] = dict()            
        simulator.data_manager.extensions['storage_to_grid_RT'] = dict()  
        simulator.data_manager.extensions['hybrid_sys_bid_RT'] = dict() 

            # 4. DA market signal (RUC) tracked quantities    
        
        simulator.data_manager.extensions['wind_to_grid_DA_ruc_tracked'] = dict()  
        simulator.data_manager.extensions['wind_to_grid_RT_ruc_tracked'] = dict()
        simulator.data_manager.extensions['wind_to_storage_DA_ruc_tracked'] = dict() 
        simulator.data_manager.extensions['wind_to_storage_RT_ruc_tracked'] = dict()        
        simulator.data_manager.extensions['storage_to_grid_DA_ruc_tracked'] = dict()
        simulator.data_manager.extensions['storage_to_grid_RT_ruc_tracked'] = dict()
        simulator.data_manager.extensions['hybrid_sys_bid_DA_ruc_tracked'] = dict()
        simulator.data_manager.extensions['hybrid_sys_bid_RT_ruc_tracked'] = dict()

            # 5. RT market signal (SCED) tracked quantities 
    
        simulator.data_manager.extensions['wind_to_grid_RT_sced_tracked'] = dict() 
        simulator.data_manager.extensions['wind_to_storage_RT_sced_tracked'] = dict()            
        simulator.data_manager.extensions['storage_to_grid_RT_sced_tracked'] = dict()  
        simulator.data_manager.extensions['hybrid_sys_bid_RT_sced_tracked'] = dict() 
        simulator.data_manager.extensions['storage_soc_sced_tracked'] = dict()

            # 6. Cleared LMPs

        simulator.data_manager.extensions['ruc_LMP'] = dict() 
        simulator.data_manager.extensions['sced_LMP'] = dict() 

        ## Bidding (in both DA and RT markets) model

        if plugin_options.bidding:            
        
            ## Get the bidding model

            print("In building bidding model")  
            
            bidding_horizon = bidding_config["bidding_horizon"]

            bidding_model = bidder.build_bidding_model(bidding_horizon)
        
            simulator.data_manager.extensions['bidding_model'] = bidding_model

            print("Bidding model is built.")
                           
        ## Tracking (both DA and RT market signals, RUC and SCED) models

        if plugin_options.track_ruc_signal:

            print("In building RUC tracking model")

            tracking_horizon = options.ruc_horizon   # plugin_options.track_horizon
            
            bidding_model_ruc = bidder.build_bidding_model(tracking_horizon)
            
            ## Get the tracking model

            ruc_tracking_model = tracker.build_tracking_model(bidding_model=bidding_model_ruc, tracking_type='RUC')            

            simulator.data_manager.extensions['ruc_tracking_model'] = ruc_tracking_model

            print("RUC tracking model is built.")
            
        if plugin_options.track_sced_signal:

            print("In building SCED tracking model")

            tracking_horizon = options.sced_horizon

            bidding_model_sced = bidder.build_bidding_model(tracking_horizon)
           
            ## Get the tracking model

            sced_tracking_model = tracker.build_tracking_model(bidding_model=bidding_model_sced, tracking_type='SCED')            

            simulator.data_manager.extensions['sced_tracking_model'] = sced_tracking_model

            print("SCED tracking model is built.")

    context.register_initialization_callback(initialize_plugin)


    def update_ruc_before_solve(options, simulator, ruc_instance, ruc_date, ruc_hour):
        """
        Solve the bidding problem and 
        pass the DA bid as pmax to the the RTS_GMLC wind generator in Prescient
        """

        _print_msg(f"In update_ruc_before_solve")
        _print_msg(f"ruc_date {ruc_date}")
        _print_msg(f"ruc_hour {ruc_hour}")

        if plugin_options.bidding:      

            ## Solve the bidding model (for the first simulation day) 

            first_date = str(dateutil.parser.parse(str(options.start_date)).date()) 

            assert first_date == bidding_date

            g = plugin_options.bidding_generator[0] 

            bidding_solution = bidder.compute_bids(simulator.data_manager.extensions['bidding_model'])

            ## Get the values of each decision variable

            wind_power_produced_DA, wind_power_produced_RT,\
            wind_to_grid_DA, wind_to_grid_RT,\
            wind_to_storage_DA, wind_to_storage_RT,\
            storage_to_grid_DA, storage_to_grid_RT, storage_soc,\
            bid_power_DA, bid_power_RT, revenue = bidding_solution  

            _print_msg(f"DA bid power: {bid_power_DA[g]}")         
            
            ## Record the bidding results
                            
            simulator.data_manager.extensions['wind_produced_DA'][g] = wind_power_produced_DA[g] 
            simulator.data_manager.extensions['wind_produced_RT'][g] = wind_power_produced_RT[g]       
            simulator.data_manager.extensions['wind_to_grid_DA'][g] = wind_to_grid_DA[g] 
            simulator.data_manager.extensions['wind_to_grid_RT'][g] = wind_to_grid_RT[g] 
            simulator.data_manager.extensions['wind_to_storage_DA'][g] = wind_to_storage_DA[g]   
            simulator.data_manager.extensions['wind_to_storage_RT'][g] = wind_to_storage_RT[g]       
            simulator.data_manager.extensions['storage_to_grid_DA'][g] = storage_to_grid_DA[g]
            simulator.data_manager.extensions['storage_to_grid_RT'][g] = storage_to_grid_RT[g]   
            simulator.data_manager.extensions['hybrid_sys_bid_DA'][g] = bid_power_DA[g]
            simulator.data_manager.extensions['hybrid_sys_bid_RT'][g] = bid_power_RT[g]

            ## Assign the hybrid system DA bid output to the generator pmax

            g_dict = ruc_instance.data['elements']['generator'][g]

            for t in range(len(bid_power_DA[g])):
                if bid_power_DA[g][t] < 0. and math.isclose(bid_power_DA[g][t], 0., abs_tol=1e-5):
                    bid_power_DA[g][t] = 0.
                                   
            g_dict['p_max']['values'][0:bidding_config['bidding_horizon']] = bid_power_DA[g] 

            _print_msg(f"Pmax in RUC: {g_dict['p_max']['values']}")           

            ## Export the bidding results to disc

            bidding_solution_dict = {'Wind DA': wind_power_produced_DA[bidding_config['gen_name'][0]],
                                     'Wind RT': wind_power_produced_RT[bidding_config['gen_name'][0]],
                                     'Wind to Grid DA': wind_to_grid_DA[bidding_config['gen_name'][0]],
                                     'Wind to Grid RT': wind_to_grid_RT[bidding_config['gen_name'][0]],                         
                                     'Wind to Storage DA': wind_to_storage_DA[bidding_config['gen_name'][0]],
                                     'Wind to Storage RT': wind_to_storage_RT[bidding_config['gen_name'][0]],                         
                                     'Storage to Grid DA': storage_to_grid_DA[bidding_config['gen_name'][0]],
                                     'Storage to Grid RT': storage_to_grid_RT[bidding_config['gen_name'][0]],                         
                                     'Hybrid Sys Bid DA': bid_power_DA[bidding_config['gen_name'][0]],
                                     'Hybrid Sys Bid RT': bid_power_RT[bidding_config['gen_name'][0]],
                                     'Storage SOC': storage_soc[bidding_config['gen_name'][0]]}

            bidding_solution_df = pd.DataFrame(bidding_solution_dict)  

            bidding_solution_df.to_csv(current_dir + '/bidding_and_tracking_outputs' + "/bidding_output_details.csv")

        else:
            return

    context.register_before_ruc_solve_callback(update_ruc_before_solve)


    def after_ruc(options, simulator, ruc_plan, ruc_date, ruc_hour):

        _print_msg(f"In after_ruc")       
        _print_msg(f"ruc_date {ruc_date}")
        _print_msg(f"ruc_hour {ruc_hour}") 

        ## Get the RUC dispatch level   

        if not plugin_options.track_ruc_signal:
            return
        ruc_instance = ruc_plan.deterministic_ruc_instance  
        
        g = plugin_options.bidding_generator[0]  
        ruc_dispatch_level = {}

        g_dict = ruc_instance.data['elements']['generator'][g] 
        ruc_dispatch_level[g] = g_dict['pg']['values']

        _print_msg(f"ruc_dispatch_level: {ruc_dispatch_level}")  

        simulator.data_manager.extensions['ruc_dispatch_level'] = ruc_dispatch_level

        ruc_schedule = simulator.data_manager.extensions['ruc_schedule']
                  
        ruc_schedule[g,ruc_date]= ruc_dispatch_level[g][0:bidding_config["bidding_horizon"]]  # RUC for the 1st day  # options.ruc_horizon

        _print_msg(f"ruc schedule for the 1st day: {ruc_schedule}")  

        ## Get the cleared DA LMPs
            # ruc_market.day_ahead_prices : { (bus, time) : price at bus for time in [0, ..., ruc_horizon-1]}
            # ruc_market.day_ahead_reserve_prices : { t : reserve price at t }

        ruc_market = ruc_plan.ruc_market

        ruc_LMP = {}
        ruc_LMP[g] = [ruc_market.day_ahead_prices[(bus_name, t)] for t in range(options.ruc_horizon)]     
               
        _print_msg(f"Cleared DA LMPs: {ruc_LMP}")    

        simulator.data_manager.extensions['ruc_LMP'][g] = ruc_LMP    

        ## Track the RUC signal

        ruc_tracking_model = simulator.data_manager.extensions['ruc_tracking_model']
       
        tracking_solution = tracker.pass_market_dispatch_and_price_to_track(ruc_tracking_model, ruc_dispatch_level, ruc_LMP, sced_dispatch=None, sced_lmp=None)
      
        wind_power_produced_DA, wind_power_produced_RT,\
        wind_to_grid_DA_ruc_tracked, wind_to_grid_RT_ruc_tracked,\
        wind_to_storage_DA_ruc_tracked, wind_to_storage_RT_ruc_tracked,\
        storage_to_grid_DA_ruc_tracked, storage_to_grid_RT_ruc_tracked, storage_soc_ruc_tracked,\
        bid_power_DA_ruc_tracked, bid_power_RT_ruc_tracked, revenue_ruc_tracked  = tracking_solution

        _print_msg(f"DA bid power ruc tracked: {bid_power_DA_ruc_tracked[g]}") 

        _print_msg(f"RT bid power ruc tracked: {bid_power_RT_ruc_tracked[g]}")       
        
        ## Record the tracked bidding results
        
        simulator.data_manager.extensions['wind_to_grid_DA_ruc_tracked'][g] = wind_to_grid_DA_ruc_tracked[g]  
        simulator.data_manager.extensions['wind_to_grid_RT_ruc_tracked'][g] = wind_to_grid_RT_ruc_tracked[g] 
        simulator.data_manager.extensions['wind_to_storage_DA_ruc_tracked'][g] = wind_to_storage_DA_ruc_tracked[g] 
        simulator.data_manager.extensions['wind_to_storage_RT_ruc_tracked'][g] = wind_to_storage_RT_ruc_tracked[g]       
        simulator.data_manager.extensions['storage_to_grid_DA_ruc_tracked'][g] = storage_to_grid_DA_ruc_tracked[g]
        simulator.data_manager.extensions['storage_to_grid_RT_ruc_tracked'][g] = storage_to_grid_RT_ruc_tracked[g]
        simulator.data_manager.extensions['hybrid_sys_bid_DA_ruc_tracked'][g] = bid_power_DA_ruc_tracked[g]
        simulator.data_manager.extensions['hybrid_sys_bid_RT_ruc_tracked'][g] = bid_power_RT_ruc_tracked[g]
            
        ## Export the tracked results to disc
  
        bidding_solution_after_ruc_dict = \
                                {'Wind DA': simulator.data_manager.extensions['wind_produced_DA'][g],
                                 'Wind RT': simulator.data_manager.extensions['wind_produced_RT'][g],
                                 'Wind to Grid DA After RUC': wind_to_grid_DA_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],
                                 'Wind to Grid RT After RUC': wind_to_grid_RT_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],  
                                 'Wind to Storage DA After RUC': wind_to_storage_DA_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],
                                 'Wind to Storage RT After RUC': wind_to_storage_RT_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],     
                                 'Storage to Grid DA After RUC': storage_to_grid_DA_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],
                                 'Storage to Grid RT After RUC': storage_to_grid_RT_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],    
                                 'Hybrid Sys Bid DA After RUC': bid_power_DA_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],
                                 'Hybrid Sys Bid RT After RUC': bid_power_RT_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']],
                                 'RUC Dispatch': list(ruc_schedule.values())[0],
                                 'Storage SOC After RUC': storage_soc_ruc_tracked[bidding_config['gen_name'][0]][0:bidding_config['bidding_horizon']]}

        bidding_solution_after_ruc_df = pd.DataFrame(bidding_solution_after_ruc_dict)  

        bidding_solution_after_ruc_df.to_csv(current_dir + '/bidding_and_tracking_outputs' + "/ruc_tracking_output_details.csv")

            
    context.register_after_ruc_generation_callback(after_ruc)


    def after_ruc_activation(options, simulator):

        _print_msg(f"In after_ruc_activation")        

        simulator.data_manager.extensions['ruc_dispatch_level_current'] =\
            simulator.data_manager.extensions['ruc_dispatch_level']

        _print_msg(f"RUC dispatch level current: {simulator.data_manager.extensions['ruc_dispatch_level_current']}")

    context.register_after_ruc_activation_callback(after_ruc_activation)


    def update_sced_before_solve(options, simulator, sced_instance): 
        """
        Solve the RUC tracking problem and 
        pass the RT bid as pmax to the the RTS_GMLC wind generator in Prescient
        """ 

        _print_msg(f"In update_sced_before_solve")

        current_time = simulator.time_manager.current_time  
        date = current_time.date  
        h = current_time.hour 
        g = plugin_options.bidding_generator[0]

        ruc_tracked_rt_power = {}
        ruc_tracked_rt_power_for_current_sced = {}
        ruc_tracked_rt_power[g] = simulator.data_manager.extensions['hybrid_sys_bid_RT_ruc_tracked'][g]
        ruc_tracked_rt_power_for_current_sced[g] = ruc_tracked_rt_power[g][h:h+options.sced_horizon]

        _print_msg(f"ruc_tracked_rt_power_for_current_sced: {ruc_tracked_rt_power_for_current_sced}")

        g_dict = sced_instance.data['elements']['generator'][g]

        for t in range(len(ruc_tracked_rt_power_for_current_sced[g])):
            if ruc_tracked_rt_power_for_current_sced[g][t] < 0. and math.isclose(ruc_tracked_rt_power_for_current_sced[g][t], 0., abs_tol=1e-5):
                ruc_tracked_rt_power_for_current_sced[g][t] = 0.

        g_dict['p_max']['values'] = ruc_tracked_rt_power_for_current_sced[g]

        _print_msg(f"current time: {current_time}")  
        _print_msg(f"Pmax in this SCED: {g_dict['p_max']['values']}") 
        _print_msg(f"Power bid submitted to RTM: {ruc_tracked_rt_power_for_current_sced[g][0]}")
            
    context.register_before_operations_solve_callback(update_sced_before_solve)


    def after_sced(options, simulator, sced_instance, lmp_instance):

        _print_msg(f"In after_sced")       

        current_time = simulator.time_manager.current_time    
        date = current_time.date
        h = current_time.hour   
        g = plugin_options.bidding_generator[0] 

        _print_msg(f"time: {current_time}")
        _print_msg(f"date: {date}")
        _print_msg(f"hour: {h}") 
        
        sced_dispatch_level = {}        
        g_dict = sced_instance.data['elements']['generator'][g]
        sced_dispatch_level[g] = g_dict['pg']['values']
                        
        _print_msg(f"SCED disptach for the current period: {sced_dispatch_level}")

        sced_schedule  = simulator.data_manager.extensions['sced_schedule']          
        sced_schedule[g,h] = sced_dispatch_level[g][0]
        _print_msg(f"SCED schedule: {sced_schedule}") 
                
        ## Get the cleared RT LMPs
        
        sced_market = lmp_instance.data["elements"]["bus"]
        sced_LMP = sced_market[bus_name]["lmp"]["values"]        
        _print_msg(f"Cleared RT LMPs (SCED LMP) for the current period: {sced_LMP}")    

        simulator.data_manager.extensions['sced_LMP'][g,h] = sced_LMP[0]    
        _print_msg(f"SCED LMPs: {simulator.data_manager.extensions['sced_LMP']}") 

        ## Track the SCED signal

        ruc_tracked_rt_power = {}
        ruc_tracked_rt_power_for_current_sced = {}
        ruc_tracked_rt_power[g] = simulator.data_manager.extensions['hybrid_sys_bid_RT_ruc_tracked'][g]
        ruc_tracked_rt_power_for_current_sced[g] = ruc_tracked_rt_power[g][h:h+options.sced_horizon]

        sced_tracking_model = simulator.data_manager.extensions['sced_tracking_model']
        ruc_LMP = simulator.data_manager.extensions['ruc_LMP'][g]
       
        tracking_solution = \
            tracker.pass_market_dispatch_and_price_to_track(sced_tracking_model, ruc_tracked_rt_power_for_current_sced, ruc_LMP, \
                                                            sced_dispatch=sced_dispatch_level, sced_lmp=sced_LMP)

        wind_power_produced_DA, wind_power_produced_RT,\
        wind_to_grid_DA_sced_tracked, wind_to_grid_RT_sced_tracked,\
        wind_to_storage_DA_sced_tracked, wind_to_storage_RT_sced_tracked,\
        storage_to_grid_DA_sced_tracked, storage_to_grid_RT_sced_tracked, storage_soc_sced_tracked,\
        bid_power_DA_sced_tracked, bid_power_RT_sced_tracked, revenue_sced_tracked  = tracking_solution

        _print_msg(f"RT bid power sced tracked for the current period: {bid_power_RT_sced_tracked[g]}") 

        ## Record the tracked bidding results

        simulator.data_manager.extensions['wind_to_grid_RT_sced_tracked'][g,h] = wind_to_grid_RT_sced_tracked[g][0]
        simulator.data_manager.extensions['wind_to_storage_RT_sced_tracked'][g,h] = wind_to_storage_RT_sced_tracked[g][0]            
        simulator.data_manager.extensions['storage_to_grid_RT_sced_tracked'][g,h] = storage_to_grid_RT_sced_tracked[g][0] 
        simulator.data_manager.extensions['hybrid_sys_bid_RT_sced_tracked'][g,h] = bid_power_RT_sced_tracked[g][0]
        simulator.data_manager.extensions['storage_soc_sced_tracked'][g,h] = storage_soc_sced_tracked[g][0]

        _print_msg(f"simulator.data_manager.extensions['hybrid_sys_bid_RT_sced_tracked']: {simulator.data_manager.extensions['hybrid_sys_bid_RT_sced_tracked']}") 

        ## Export the tracked results to disc

        if h == bidding_config["bidding_horizon"]-1:           
                    
            bidding_solution_after_sced_dict = \
                                    {'Wind DA': simulator.data_manager.extensions['wind_produced_DA'][g],
                                     'Wind RT': simulator.data_manager.extensions['wind_produced_RT'][g],                                 
                                     'Wind to Grid RT After SCED': list(simulator.data_manager.extensions['wind_to_grid_RT_sced_tracked'].values()),                                
                                     'Wind to Storage RT After SCED': list(simulator.data_manager.extensions['wind_to_storage_RT_sced_tracked'].values()),                                
                                     'Storage to Grid RT After SCED': list(simulator.data_manager.extensions['storage_to_grid_RT_sced_tracked'].values()),
                                     'Hybrid Sys Bid RT After SCED': list(simulator.data_manager.extensions['hybrid_sys_bid_RT_sced_tracked'].values()),
                                     'SCED Dispatch': list(sced_schedule.values()),
                                     'Storage SOC After SCED': list(simulator.data_manager.extensions['storage_soc_sced_tracked'].values())}

            bidding_solution_after_sced_df = pd.DataFrame(bidding_solution_after_sced_dict)  

            bidding_solution_after_sced_df.to_csv(current_dir + '/bidding_and_tracking_outputs' + "/sced_tracking_output_details.csv")       

    context.register_after_operations_callback(after_sced)
    

    def update_observed_wind_dispatch(options, simulator, ops_stats):
       
        _print_msg(f"In update_observed_wind_dispatch")        

        current_time = simulator.time_manager.current_time
        d = current_time.date  
        h = current_time.hour  
        g = plugin_options.bidding_generator[0]
        
        if plugin_options.track_ruc_signal:

            print('Making changes in observed power output using tracking RUC model.') 

            ops_stats.observed_renewables_levels[g] = simulator.data_manager.extensions['ruc_schedule'][g,d][h]
            observed_dispatch = ops_stats.observed_renewables_levels[g]
            _print_msg(f"RUC observed wind dispatch at {g} {observed_dispatch}")

            observed_lmp = ops_stats.observed_bus_LMPs
            _print_msg(f"Observed RUC LMP at {g} {observed_lmp[bus_name]}")

        if plugin_options.track_sced_signal:

            print('Making changes in observed power output using tracking SCED model.') 

            ops_stats.observed_renewables_levels[g] = simulator.data_manager.extensions['sced_schedule'][g,h]
            observed_dispatch = ops_stats.observed_renewables_levels[g]
            _print_msg(f"SCED observed wind dispatch at {g} {observed_dispatch}")

            observed_lmp = ops_stats.observed_bus_LMPs
            _print_msg(f"Observed SCED LMP at {g} {observed_lmp[bus_name]}")

    context.register_update_operations_stats_callback(update_observed_wind_dispatch)