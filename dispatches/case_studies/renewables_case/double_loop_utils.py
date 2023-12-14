#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
import numpy as np
import pandas as pd
from pathlib import Path
import utm
import networkx as nx


def read_prescient_file(filepath):
    df = pd.read_csv(filepath)
    if 'Minute' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype('str') + " " + df['Hour'].astype('str') + ":" + df['Minute'].astype('str'), format='%Y-%m-%d %H:%M')
        df = df.drop(['Date', 'Hour', 'Minute'], axis=1)
    else:
        df['Datetime'] = pd.to_datetime(df['Date'].astype('str') + " " + df['Hour'].astype('str') + ":00", format='%Y-%m-%d %H:%M')
        df = df.drop(['Date', 'Hour'], axis=1)
    
    df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    df = df.drop(['Datetime'], axis=1)
    return df


def read_prescient_outputs(output_dir, source_dir, gen_name=None):
    """
    Read the bus information, bus LMPs, and renewables output detail CSVs from Prescient simulations

    Args:
        output_dir: Prescient simulation output directory
        source_dir: "SourceData" folder of the RTS-GMLC
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    summary = read_prescient_file(output_dir / "hourly_summary.csv")
    
    bus_detail_df = read_prescient_file(output_dir / "bus_detail.csv")
    bus_detail_df['LMP'] = bus_detail_df['LMP'].astype('float64')
    bus_detail_df['LMP DA'] = bus_detail_df['LMP DA'].astype('float64')
    bus_detail_df = bus_detail_df.rename(columns={'Demand': 'BusDemand', 'Overgeneration': 'BusOvergeneration'})

    summary = pd.merge(summary, bus_detail_df, how='outer', on=['Datetime'])

    renewables_df = read_prescient_file(output_dir / "renewables_detail.csv")
    thermal_df = read_prescient_file(output_dir / 'thermal_detail.csv')

    load_renewables = gen_name == None or gen_name in renewables_df.Generator.unique()
    load_thermal = gen_name == None or gen_name in thermal_df.Generator.unique()
    
    if load_renewables:
        gen_df = renewables_df

    if load_thermal:
        gen_df = thermal_df

    if load_thermal and load_renewables:
        gen_df = pd.merge(renewables_df, thermal_df, how='outer', on=['Datetime', 'Generator', 'Unit Market Revenue', 'Unit Uplift Payment'])

    return summary, gen_df


def read_rts_gmlc_wind_inputs_with_fix(source_dir, gen_df, aggfunc='first'):
    """
    For simulations that don't start at 01-01, the time index of the CF here may be shifted by one day relative to the output reported from renewables_detail.csv.
    To check that the time indices are correct, compare the DA output with the CF and make sure that the output never exceeds the CF.

    Use '317_WIND_1' to checkt he issue with the timestep. If the check passes with 317_wind_1 it should pass for the others as well.

    """
    gen_317_WIND_1='317_WIND_1'
    pmax_317_WIND_1 = 799.1

    gen_wind = gen_df[gen_df['Generator'] == gen_317_WIND_1]

    wind_df = read_rts_gmlc_wind_inputs(source_dir, None, aggfunc)

    if len(wind_df) == len(gen_wind):
        assert (wind_df.index == gen_wind.index).all()
        wind_cfs = wind_df
        # DA cfs should have no problems
        if not (wind_cfs[f'{gen_317_WIND_1}-RTCF'] * pmax_317_WIND_1 - gen_wind['Output']).min() > -1e-4:
            # try to fix by changing the agg func
            wind_cfs = read_rts_gmlc_wind_inputs(source_dir, gen_317_WIND_1, agg_func='mean')
    else:
        wind_cfs = wind_df[wind_df.index.isin(gen_wind.index)]
        if not (wind_cfs[f'{gen_317_WIND_1}-DACF'] * pmax_317_WIND_1 - gen_wind['Output DA']).min() > -1e-4:
            # try shifting the time index
            for c in wind_df.columns:
                wind_df[c] = np.roll(wind_df[c].values, 1)
            wind_cfs = wind_df[wind_df.index.isin(gen_wind.index)]
        if not (wind_cfs[f'{gen_317_WIND_1}-RTCF'] * pmax_317_WIND_1 - gen_wind['Output']).min() > -1e-4:
            # try to fix by changing the agg func
            if aggfunc != 'mean':
                wind_cfs = read_rts_gmlc_wind_inputs_with_fix(source_dir, gen_df, aggfunc='mean')

    assert (wind_cfs[f'{gen_317_WIND_1}-DACF'] * pmax_317_WIND_1 - gen_wind['Output DA']).min() > -1e-4
    assert (wind_cfs[f'{gen_317_WIND_1}-RTCF'] * pmax_317_WIND_1 - gen_wind['Output']).min() > -1e-4

    return wind_cfs
    

def read_rts_gmlc_wind_inputs(source_dir, gen_name=None, agg_func="first"):
    """
    Read capacity factors for day ahead and real time wind forecasts. 
    The forecasts are provided as 12 periods per hour, or 288 per day, and are aggregated to hourly data. 
    The aggregation can be done by taking the mean of the 12 periods, or taking the first. Check your Prescient parameters or outputs to verify.

    Warnings: 

    For simulations that don't start at 01-01, the time index of the CF here may be shifted by one day relative to the output reported from renewables_detail.csv.
    To check that the time indices are correct, compare the DA output with the CF and make sure that the output never exceeds the CF.

    Use `read_rts_gmlc_wind_inputs_with_fix` to check and try to fix these issues

    Args:
        source_dir: "SourceData" folder of the RTS-GMLC
        gen_name: optional wind generator name, if not provided then all wind generators returned
    """
    source_dir = Path(source_dir)
    gen_df = pd.read_csv(source_dir / "gen.csv")
    if not gen_name:
        wind_gens = [i for i in gen_df['GEN UID'] if "WIND" in i]
    else:
        wind_gens = [gen_name]

    wind_rt_df = pd.read_csv(source_dir.parent / "timeseries_data_files" / "WIND" / "REAL_TIME_wind.csv")
    wind_da_df = pd.read_csv(source_dir.parent / "timeseries_data_files" / "WIND" / "DAY_AHEAD_wind.csv")

    start_df = wind_rt_df.head(1)
    start_year = start_df.Year.values[0]
    start_mon = start_df.Month.values[0]
    start_day = start_df.Day.values[0]

    start_date = pd.Timestamp(f"{start_year}-{start_mon:02d}-{start_day:02d} 00:00:00")
    ix = pd.date_range(start=start_date, 
                        end=start_date
                        + pd.offsets.DateOffset(days=366)
                        - pd.offsets.DateOffset(hours=1),
                        freq='1H')
    wind_df = pd.DataFrame(index=ix)
    for k in wind_gens:
        rt_wind = wind_rt_df[k].values
        rt_wind = np.reshape(rt_wind, (8784, 12))
        if agg_func == "first":
            rt_wind = rt_wind[:, 0]
        elif agg_func == "mean":
            rt_wind = rt_wind.mean(1)
        else:
            raise ValueError(f"Unrecognized agg_func {agg_func}. Options are 'first' or 'mean'.")
        da_wind = wind_da_df[k].values

        wind_pmax = gen_df[gen_df['GEN UID'] == k]['PMax MW'].values[0]
        wind_df[k+"-RTCF"] = rt_wind / wind_pmax
        wind_df[k+"-DACF"] = da_wind / wind_pmax
    return wind_df


def get_rtsgmlc_bus_dict(source_dir):
    source_dir = Path(source_dir)
    bus_names = pd.read_csv(source_dir / "bus.csv")
    bus_dict = {k: v for k, v in zip(bus_names['Bus ID'].values, bus_names['Bus Name'].values)}
    return bus_dict


def prescient_outputs_for_gen(output_dir, source_dir, gen_name):
    """
    Get timeseries RT and DA Outputs and Capacity factors, Curtailment, Unit Market Revenue, Unit Uplift Payment for the given generator,
    and also the Demand, Shortfall, Overgeneration, and LMPs for the bus of the generator

    Args:
        output_dir: Prescient simulation output directory
        source_dir: "SourceData" folder of the RTS-GMLC
        gen_name: optional wind generator name, if not provided then all wind generators returned
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    summary, gen_df = read_prescient_outputs(output_dir, source_dir, gen_name)
    if "WIND" in gen_name or "PV" in gen_name:
        # double loop may have set the wind or pv plant as a thermal generator, so then these columns are not meaningful
        summary = summary.drop(['RenewablesUsed', 'RenewablesCurtailment'], axis=1)

    bus_dict = get_rtsgmlc_bus_dict(source_dir)
    bus_name = bus_dict[int(gen_name.split('_')[0])]

    summary = summary[summary.Bus == bus_name]
    gen_df = gen_df[gen_df.Generator == gen_name]
    df = pd.concat([summary, gen_df], axis=1)

    if 'WIND' in gen_name:
        wind_forecast_df = read_rts_gmlc_wind_inputs(source_dir, gen_name)
        wind_forecast_df = wind_forecast_df[wind_forecast_df.index.isin(df.index)]
        df = pd.concat([df, wind_forecast_df], axis=1)
    return df


def prescient_double_loop_outputs_for_gen(output_dir):
    """
    Get timeseries double-loop simulation outputs
    """
    output_dir = Path(output_dir)
    tracker_df = read_prescient_file(output_dir / "tracker_detail.csv")

    tracker_model_df = read_prescient_file(output_dir / "tracking_model_detail.csv")
    gen_name = tracker_model_df['Generator'].values[0]
    tracker_model_df = tracker_model_df.drop(['Generator'], axis=1)

    tracker_df = pd.merge(tracker_df, tracker_model_df, how='left', on=['Datetime', 'Horizon [hr]'])
    tracker_df.loc[:, 'Model'] = 'Tracker'

    bidder_df = pd.read_csv(output_dir / "bidder_detail.csv")
    da_bidder_df = bidder_df[bidder_df['Market'] == 'Day-ahead']
    da_bidder_df = da_bidder_df.rename(columns={'Hour': 'Horizon [hr]', 'Market': 'Model', 'Power 0 [MW]': 'DA Power 0 [MW]', 'Cost 0 [$]': 'DA Cost 0 [$]', 'Power 1 [MW]': 'DA Power 1 [MW]',
       'Cost 1 [$]': 'DA Cost 1 [$]', 'Power 2 [MW]': 'DA Power 2 [MW]', 'Cost 2 [$]': 'DA Cost 2 [$]'})
    da_bidder_df.loc[:, 'Model'] = 'DA Bidder'
    da_bidder_df['Datetime'] = pd.to_datetime(da_bidder_df['Date'].astype('str') + " 00:00:00", format='%Y-%m-%d %H:%M:%S')
    da_bidder_df = da_bidder_df.set_index(pd.DatetimeIndex(da_bidder_df['Datetime']))
    da_bidder_df = da_bidder_df.drop(['Date', 'Datetime', 'Generator'], axis=1)

    rt_bidder_df = bidder_df[bidder_df['Market'] == 'Real-time']
    rt_bidder_df = rt_bidder_df.rename(columns={'Market': 'Model', 'Power 0 [MW]': 'RT Power 0 [MW]', 'Cost 0 [$]': 'RT Cost 0 [$]', 'Power 1 [MW]': 'RT Power 1 [MW]',
       'Cost 1 [$]': 'RT Cost 1 [$]', 'Power 2 [MW]': 'RT Power 2 [MW]', 'Cost 2 [$]': 'RT Cost 2 [$]'})
    rt_bidder_df.loc[:, 'Model'] = 'RT Bidder'
    rt_bidder_df['Hour'] = np.tile(np.repeat(range(24), 4), 366)[0:len(rt_bidder_df)]
    rt_bidder_df['Horizon [hr]'] = np.tile(range(4), 8784)[0:len(rt_bidder_df)]
    rt_bidder_df['Datetime'] = pd.to_datetime(rt_bidder_df['Date'].astype('str') + " " + rt_bidder_df['Hour'].astype('str') + ":00", format='%Y-%m-%d %H:%M')
    rt_bidder_df = rt_bidder_df.set_index(pd.DatetimeIndex(rt_bidder_df['Datetime']))
    rt_bidder_df = rt_bidder_df.drop(['Date', 'Hour', 'Datetime', 'Generator'], axis=1)

    bidder_df = pd.merge(da_bidder_df, rt_bidder_df, how='outer', on=['Datetime', 'Horizon [hr]', 'Model'])
    df = pd.merge(bidder_df, tracker_df, how='outer', on=['Datetime', 'Horizon [hr]', 'Model'])
    return df, gen_name


def double_loop_outputs_for_gen(double_loop_dir, source_dir):
    source_dir = Path(source_dir)
    double_loop_dir = Path(double_loop_dir)
    dl_df, gen_name = prescient_double_loop_outputs_for_gen(double_loop_dir)
    res_df = prescient_outputs_for_gen(double_loop_dir, source_dir, gen_name)
    res_df = res_df[res_df.index.isin(dl_df.index.unique())]
    res_df['Datetime'] = res_df.index
    res_df.loc[:, 'Model'] = "Prescient"

    df = pd.merge(dl_df, res_df, how='outer', on=['Datetime', 'Model']).set_index('Datetime')
    df.Model = pd.Categorical(df.Model, categories=['DA Bidder', 'RT Bidder', 'Tracker', "Prescient"])
    df = df.sort_values(by = ['Datetime', 'Model'], na_position = 'last')
    return df.dropna(axis=1, how='all')


def get_rtsgmlc_network(output_dir, source_dir, length_scaling=1/300, width_scaling=1/75):
    edges_df = pd.read_csv(source_dir / "branch.csv").set_index("UID")
    nodes_df = pd.read_csv(source_dir / "bus.csv").set_index("Bus ID")
    nodes_df.index = nodes_df.index.map(str)
    line_detail = read_prescient_file(output_dir / "line_detail.csv")
    summary, gen_df = read_prescient_outputs(output_dir, source_dir)
    bus_dict = {k: v for k, v in zip(nodes_df.index.values, nodes_df['Bus Name'].values)}

    # Create Network with Edge info
    cont_rating = edges_df['Cont Rating'].values.tolist()
    line_detail['Cont Rating'] = cont_rating * len(line_detail.index.unique())
    line_detail["Relative Flow"] = np.abs(line_detail['Flow'] / line_detail['Cont Rating'])

    G = nx.from_pandas_edgelist(edges_df.reset_index().astype('str'), source='From Bus', target='To Bus', edge_attr=('Cont Rating', 'LTE Rating', 'STE Rating', 'Length', "UID"))

    # Fill in Node X, Y info
    nodes_df['x'], nodes_df['y'] = utm.from_latlon(nodes_df['lat'].values, nodes_df['lng'].values)[0:2]

    x_pos = (nodes_df['x'] - nodes_df['x'].mean()) * length_scaling
    y_pos = (nodes_df['y'] - nodes_df['y'].mean()) * length_scaling
    nx.set_node_attributes(G, x_pos.to_dict(), name="x")
    nx.set_node_attributes(G, y_pos.to_dict(), name="y")
    nx.set_node_attributes(G, nodes_df['Bus Name'].to_dict(), name="name")

    # Fill in Node colors by generators
    color_dict = {}
    wind_buses = gen_df[gen_df['Generator'].str.contains('WIND')]['Generator'].str.slice(0, 3).unique()
    pv_buses = gen_df[gen_df['Generator'].str.contains('PV')]['Generator'].str.slice(0, 3).unique()
    nuc_buses = gen_df[gen_df['Generator'].str.contains('NUCLEAR')]['Generator'].str.slice(0, 3).unique()

    for n in G.nodes:
        if n in wind_buses and n in pv_buses and n in nuc_buses:
            color = 'white'
        elif n in wind_buses and n in pv_buses:
            color = 'green'
        elif n in wind_buses and n in nuc_buses:
            color = "purple"
        elif n in pv_buses and n in nuc_buses:
            color = 'orange'
        elif n in wind_buses:
            color = 'blue'
        elif n in pv_buses:
            color = 'yellow'
        elif n in nuc_buses:
            color = "red"
        else:
            color = 'grey'
        color_dict[n] = color
    nx.set_node_attributes(G, color_dict, name="color")

    # Add Edge Width for Rating and congestion for color
    line_detail['Relative OverFlow'] = (line_detail['Relative Flow'] >= 0.95).astype(int) * line_detail['Relative Flow']
    edges_df['Relative OverFlow'] = line_detail.set_index('Line')['Relative OverFlow'].groupby('Line').mean()
    edges_df['Relative Flow'] = line_detail.set_index('Line')['Relative Flow'].groupby('Line').mean()
    edges_df

    avg_LMP_diff = []
    for (i, j) in edges_df[['From Bus', 'To Bus']].values:
        LMP_diff = (summary.query(f"Bus == '{bus_dict[str(i)]}'")['LMP'] - summary.query(f"Bus == '{bus_dict[str(j)]}'")['LMP']).abs()
        avg_LMP_diff.append(LMP_diff.mean())
    edges_df['Avg LMP Diff'] = avg_LMP_diff
    edges_df

    edges_dict = {}
    edge_color_val = 'Avg LMP Diff' # or 'Relative OverFlow'
    max_cong = edges_df[edge_color_val].max()
    for (i, j, w, o) in edges_df[['From Bus', 'To Bus', 'Cont Rating', edge_color_val]].values:
        if o != 0:
            rel_cong = o / max_cong
            gb =  75 + int(180 * rel_cong)
            color = '#%02x%02x%02x' % (gb, 0, 0)
        else:
            color = 'black'
    for (i, j, w, o) in edges_df[['From Bus', 'To Bus', 'Cont Rating', edge_color_val]].values:
        if o != 0:
            rel_cong = o / max_cong
            gb =  75 + int(180 * rel_cong)
            color = '#%02x%02x%02x' % (gb, 0, 0)
        else:
            color = 'black'
        edges_dict[(str(int(i)), str(int(j)))] = {'width': w * width_scaling, 'color': color, 'title': str(round(o, 2)), edge_color_val: o}
    nx.set_edge_attributes(G, edges_dict)
    return G, line_detail, edges_df, gen_df, summary