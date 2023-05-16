import copy
import os
import pathlib
import pandas as pd
import numpy as np

'''
prescient_output_to_df, get_gdf and summarize_results are modified from Ben's fork. 
Run summarize_results will give us the csv file of the simulation including DA dispatch, DA LMP, RT dispatch and RT LMP (like the sweep results)

summarize_revenue is modified from Xian's fork. This function return a dictionary of DA dispatch, DA revenue, RT dispatch and RT revenue and total revenue.

summarize_H2_revenue is to calculate the total hydrogen revenue.
'''


def prescient_output_to_df(file_name):
    '''Helper for loading data from Prescient output csv.
        Combines Datetimes into single column.
    '''
    df = pd.read_csv(file_name)
    df['Datetime'] = \
        pd.to_datetime(df['Date']) + \
        pd.to_timedelta(df['Hour'], 'hour') + \
        pd.to_timedelta(df['Minute'], 'minute')
    df.drop(columns=['Date','Hour','Minute'], inplace=True)
    # put 'Datetime' in front
    cols = df.columns.tolist()
    cols = cols[-1:]+cols[:-1]
    
    return df[cols]


def get_gdf(directory, generator_file_name, generator_name, dispatch_name):
    gdf = prescient_output_to_df(os.path.join(directory, generator_file_name))
    gdf = gdf[gdf["Generator"] == generator_name][["Datetime", dispatch_name, dispatch_name + " DA"]]
    gdf.set_index("Datetime", inplace=True)
    gdf.rename(columns={ dispatch_name : generator_name + " Dispatch", dispatch_name + " DA" : generator_name + " Dispatch DA"}, inplace=True)

    return gdf


def summarize_results(base_directory, generator_name, bus_name, output_directory):
    """
    Summarize Prescient runs for a single generator
    Args:
        base_directory (str) : the base directory name (without index)
        generator_name (str) : The generator name to get the dispatch for. Looks in thermal_gens.csv and then renewable_gens.csv.
        bus_name (str) : The bus to get the LMPs for.
        output_directory (str) : The location to write the summary files to.
    Returns:
        None
    """

    # param_file = os.path.join(output_directory, "sweep_parameters.csv")

    # figure out if renewable or thermal or virtual
    generator_file_names = ("thermal_detail.csv", "renewables_detail.csv", "virtual_detail.csv")
    dispatch_name_map = { "thermal_detail.csv" : "Dispatch",
                          "renewables_detail.csv" : "Output",
                          "virtual_detail.csv" : "Output",
                        }

    def _get_gen_df(generator_name):
        for generator_file_name in generator_file_names:
            gdf = pd.read_csv(os.path.join(base_directory, generator_file_name))["Generator"]
            if generator_name in gdf.unique():
                return generator_file_name
        else: # no break
            raise RuntimeError("Could not find output for generator "+generator_name)

    generator_file_name = _get_gen_df(generator_name)
    dispatch_name = dispatch_name_map[generator_file_name]


    # if not os.path.isfile(os.path.join(directory, "overall_simulation_output.csv")):
    #     raise Exception(f"For index {idx}, the simulation did not complete!")

    gdf = get_gdf(base_directory, generator_file_name, generator_name, dispatch_name)
    df_list = [gdf]
    RT_names = [gdf.columns[0]]
    DA_names = [gdf.columns[1]]


    bdf = prescient_output_to_df(os.path.join(base_directory, "bus_detail.csv"))
    bdf = bdf[bdf["Bus"] == bus_name][["Datetime","LMP","LMP DA"]]
    bdf.set_index("Datetime", inplace=True)
    df_list.append(bdf)
    RT_names.append(bdf.columns[0])
    DA_names.append(bdf.columns[1])

    odf = pd.concat(df_list, axis=1)[[*RT_names,*DA_names]]
    odf.to_csv(os.path.join(output_directory, f"{base_directory}.csv"))


def summarize_revenue(sim_id, result_dir, gen_detail, bus_name, gen_name):
    
    '''
    Summary the total DA and RT dispatch and revenue.
    Args:
        sim_id(int): simulation id
        result_dir(str): the result directory name
        gen_detail(str): generator results file
        bus_name(str): the bus to get the LMPs for.
        gen_name(str): name of the generator
    Returns:
        summary(dict): the total DA and RT dispatch and revenue information
    '''

    df = pd.read_csv(os.path.join(result_dir, gen_detail))
    df = df.loc[df["Generator"] == gen_name]
    df["Time Index"] = range(len(df))
    df.rename(columns={"Output": "Dispatch", "Output DA": "Dispatch DA"}, inplace=True)

    bus_df = pd.read_csv(os.path.join(result_dir, "bus_detail.csv"))
    bus_df = bus_df.loc[bus_df["Bus"] == bus_name]
    bus_df["Time Index"] = range(len(bus_df))

    df = df.merge(bus_df, how="left", left_on="Time Index", right_on="Time Index")

    df["Revenue DA"] = df["Dispatch DA"] * df["LMP DA"]
    df["Revenue RT"] = (df["Dispatch"] - df["Dispatch DA"]) * df["LMP"]
    df["Total Revenue"] = df["Revenue DA"] + df["Revenue RT"]

    df = df[["Dispatch", "Dispatch DA", "Revenue DA", "Revenue RT", "Total Revenue"]]

    summary = df.sum().to_dict()
    summary["sim_id"] = sim_id
    
    return summary


def summarize_in_df(result_dir, gen_detail, bus_name, gen_name):
    
    '''
    Summary the hourly DA and RT dispatch and LMP information.
    Args:
        result_dir(str): the result directory name
        gen_detail(str): generator results file
        bus_name(str) the bus to get the LMPs for.
        gen_name(str): name of the generator
    Returns:
        df(dataframe): the hourly DA and RT dispatch and revenue information
    '''

    df = pd.read_csv(os.path.join(result_dir, gen_detail))
    df = df.loc[df["Generator"] == gen_name]
    df["Time Index"] = range(len(df))
    df.rename(columns={"Output": "Dispatch", "Output DA": "Dispatch DA"}, inplace=True)

    bus_df = pd.read_csv(os.path.join(result_dir, "bus_detail.csv"))
    bus_df = bus_df.loc[bus_df["Bus"] == bus_name]
    bus_df["Time Index"] = range(len(bus_df))

    df = df.merge(bus_df, how="left", left_on="Time Index", right_on="Time Index")

    df["Revenue DA"] = df["Dispatch DA"] * df["LMP DA"]
    df["Revenue RT"] = (df["Dispatch"] - df["Dispatch DA"]) * df["LMP"]
    df["Total Revenue"] = df["Revenue DA"] + df["Revenue RT"]

    df = df[["Dispatch", "Dispatch DA", "Revenue DA", "Revenue RT", "Total Revenue"]]

    return df


def summarize_H2_revenue(df, PEM_size, H2_price, gen_name):
    
    '''
    Summary the H2 revenue. 
    Args:
        df(dataframe): dataframe for the hourly dispatch information (from summarize_in_df)
        PEM_size(float): the size of the PEM, MW.
        H2_price(float): Hydrogen price, $/kg
        gen_name(str): name of the generator
    Returns:
        (dict): the DA and RT dispatch and revenue information
    '''
    # read the wind data
    df_wind = pd.read_csv("Real_Time_wind_hourly.csv")[gen_name]
    
    # calculate excess elec (can be used for PEM or curtailed)
    rt_dispatch = df["Dispatch"]
    df_ph = df_wind - rt_dispatch
    excess_elec = df_ph.to_numpy()

    # Find and remove Nan indexes.
    nan_idx = np.isnan(excess_elec)
    excess_elec_without_nan = excess_elec[~nan_idx]

    # calculate the electricity that PEM used to produce H2
    pem_elec = np.clip(excess_elec_without_nan, 0, PEM_size)
    # calculate H2 revenue
    eh_rate = 54.953 # kWh/kg
    h2_revenue = sum(pem_elec)/eh_rate*H2_price*1000

    df_H2 = {}
    df_H2["Total PEM electricity"] = np.sum(pem_elec)
    df_H2["Total H2 revenue"] = h2_revenue
    return df_H2


base_directory = "new_Benchmark_single_wind_gen_sim_15_200_rth_4"
# base_directory = "new_Benchmark_wind_pem_parametrized_rf_0.15_shortfall_200_rth_1"
# base_directory = "new_Benchmark_wind_battery_stochastic_bidder_rf_0.15_shortfall_200_rth_4"
# base_directory = "new_Benchmark_wind_battery_parametrized_bidder_fix_commitment_rf_0.15_shortfall_200_rth_1"
generator_name = "303_WIND_1"
bus_name = "Caesar"
gen_detail = "renewables_detail.csv"
PEM_size = 200 # MW
H2_price = 1.25 # $/kg

# # Want hourly dispatch and LMP data
output_directory = os.getcwd()
summarize_results(base_directory, generator_name, bus_name, output_directory)

# # Want a total revenue and dispatch summary
result = summarize_revenue(1, base_directory, gen_detail, bus_name, generator_name)
print(result)

# # Want the H2 revenue information.  
# df = summarize_in_df(base_directory, gen_detail, bus_name, generator_name)
# result = summarize_H2_revenue(df, PEM_size, H2_price, generator_name)
# print(result)
