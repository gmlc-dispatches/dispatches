import os
from Simulation_Data_FE_separate import SimulationData
from tslearn.utils import to_time_series_dataset
import pathlib
import pandas as pd


def scale_data(generator_scaled_dispatch_dict, storage_scaled_dispatch_dict, index):

    '''
    Get the capacity factor from generator and storage and sum them to 1.2
    '''


    scaled_dispatch_dict = {}
    
    for idx in index:
        generator_cf =  generator_scaled_dispatch_dict[idx]
        storage_cf = storage_scaled_dispatch_dict[idx]
        sum_cf = generator_cf + storage_cf*0.2
        scaled_dispatch_dict[idx] = sum_cf

    return scaled_dispatch_dict

def main():
    os.chdir('..')
    # for FE case study
    dispatch_data_path_generator = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/Dispatch_generator_data_FE_separate_whole.csv')
    dispatch_data_path_storage = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/Dispatch_storage_data_FE_separate_whole.csv')
    input_data_path = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/sweep_parameters_results_FE_whole.h5')
    case_type = 'FE'
    num_clusters = 20
    num_sims = 400
    input_layer_node = 4
    filter_opt = True

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data_generator = SimulationData(dispatch_data_path_generator, input_data_path, num_sims, case_type)
    generator_scaled_dispatch_dict = simulation_data_generator._scale_data_generator()

    simulation_data_storage = SimulationData(dispatch_data_path_storage, input_data_path, num_sims, case_type)
    storage_scaled_dispatch_dict = simulation_data_storage._scale_data_storage()

    index = simulation_data_storage._index

    scaled_dispatch_dict = scale_data(generator_scaled_dispatch_dict, storage_scaled_dispatch_dict, index)

    # save the data to csv
    year_cf_list = []
    df_index = []
    for i in index:
        year_cf = scaled_dispatch_dict[i]
        df_year_cf = pd.DataFrame(year_cf).T
        year_cf_list.append(df_year_cf)
        df_index.append(i)

    data_output = pd.concat(year_cf_list)
    data_output.index = df_index
    data_output.to_csv('FE_separate_FIX_UC/separate_cf.csv', index = True)


if __name__ == "__main__":
    main()