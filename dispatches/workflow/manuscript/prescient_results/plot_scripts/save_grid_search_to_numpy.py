import pandas as pd
import numpy as np
import os

result_dir = 'pmax_grid_search_3/price_results'
os.makedirs(result_dir, exist_ok=True)
for index in range(0,16):
    result_data_dir = 'pmax_grid_search_3/deterministic_simulation_output_index_{}'.format(index)

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

    #number of startups
    thermal_detail_diff = gen_results['Unit State'].diff().fillna(0)
    state_changes = thermal_detail_diff.value_counts()

    if 1 in state_changes.keys():
        n_startups = state_changes[1]
    else:
        n_startups = 0 

    #Save prices, dispatch, and n_startups
    with open(result_dir+'/rts_results_grid_search_{}.npy'.format(index), 'wb') as f:
        np.save(f,dispatch)
        np.save(f,lmp)
        np.save(f,n_startups)