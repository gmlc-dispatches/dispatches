import pandas as pd
import numpy as np
import os

parm_idx = 15
base_dir = 'pmax_sweep_runs/sensitivity_sweep_over_pmax_{}'.format(parm_idx)

result_dir = base_dir+'/pmax_sweep_prices_{}'.format(parm_idx)
os.makedirs(result_dir, exist_ok=True)

p_max_vector = np.arange(175,450,25)

for index in range(0,11):
    result_data_dir = base_dir+'/deterministic_simulation_output_index_{}'.format(index)

    bus_detail_df = pd.read_csv(os.path.join(result_data_dir,'bus_detail.csv'))

    # thermal detail (power dispatched by each generator)
    thermal_detail_df = pd.read_csv(os.path.join(result_data_dir,'thermal_detail.csv'))

    # renewable details
    renewable_detail_df = pd.read_csv(os.path.join(result_data_dir,'renewables_detail.csv'))

    # line detail (this has the power flow on each line)
    line_detail_df = pd.read_csv(os.path.join(result_data_dir,'line_detail.csv'))

    # hourly summary
    hourly_summary_df = pd.read_csv(os.path.join(result_data_dir,'hourly_summary.csv'))

    # the list of unique thermal generators
    generator_list = pd.unique(thermal_detail_df['Generator'])

    # the list of unique renewable power plants
    renewable_list = pd.unique(renewable_detail_df['Generator'])

    #Look at an example coal generator (This is the generator we studied for the big Prescient simulation data set)
    coal_generator = '123_STEAM_3'
    gen_results = thermal_detail_df.loc[thermal_detail_df['Generator'] == coal_generator] #output results for this generator
    dispatch = gen_results["Dispatch"].to_numpy()

    bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == "CopperSheet"] #output results for this generator
    lmp = np.copy(bus_results["LMP"])
    lmp[lmp > 200] = 200

    #Save prices and dispatch
    
    with open(result_dir+'/rts_results_all_prices_pmax_{}.npy'.format(p_max_vector[index]), 'wb') as f:
        np.save(f,dispatch)
        np.save(f,lmp)

