import pandas as pd
import numpy as np
import os


def read_seperately(file_name):

    csv_file = file_name
    df_input_data = pd.read_csv(csv_file)
    # hdf_file = file_name + '.h5'
    # df_input_data.to_hdf(hdf_file, key = 'df_input_data')

    return df_input_data


def read_together(file_name_list):
    rf = [10,10,15,15]
    mlmp = [500,1000,500,1000]


    df_input_data_list = []
    for reserve_factor, max_lmp, file_name in zip(rf, mlmp, file_name_list):
        df_input_data = read_seperately(file_name)

        reserve_factor_array = np.array([reserve_factor]*56)
        max_lmp_array = np.array([max_lmp]*56)

        df_input_data.insert(df_input_data.shape[1],'reserve_factor', reserve_factor_array)
        df_input_data.insert(df_input_data.shape[1],'max_lmp', max_lmp_array)
        df_input_data_list.append(df_input_data)

    for idx, df in enumerate(df_input_data_list):
        for j in range(56):
            df.loc[j,'index'] = j+(idx)*56

    df_all = pd.concat(df_input_data_list, axis=0, ignore_index=True)

    df_all.to_hdf('results_renewable_sweep_Wind_H2\\sweep_parameters_results_nuclear_whole', key = 'df_all')
    pd.options.display.max_rows = None
    print(df_all)

    return


def main():
    f1 = 'results_renewable_sweep_Wind_H2\\input_data\\sweep_parameters_results_parameter_sweep_10_500.csv'
    f2 = 'results_renewable_sweep_Wind_H2\\input_data\\sweep_parameters_results_parameter_sweep_10_1000.csv'
    f3 = 'results_renewable_sweep_Wind_H2\\input_data\\sweep_parameters_results_parameter_sweep_15_500.csv'
    f4 = 'results_renewable_sweep_Wind_H2\\input_data\\sweep_parameters_results_parameter_sweep_15_1000.csv'
    file_name_list = [f1,f2,f3,f4]

    read_together(file_name_list)


if __name__ == '__main__':
    main()
