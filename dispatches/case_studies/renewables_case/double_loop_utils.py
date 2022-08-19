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
import numpy as np
import pandas as pd
from pathlib import Path

def read_prescient_outputs(output_dir, source_dir, gen_name=None):
    """
    Read the bus information, bus LMPs, and renewables output detail CSVs from Prescient simulations

    Args:
        output_dir: Prescient simulation output directory
        source_dir: "SourceData" folder of the RTS-GMLC
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    summary = pd.read_csv(output_dir / "hourly_summary.csv")
    summary['Datetime'] = pd.to_datetime(summary['Date'].astype('str') + " " + summary['Hour'].astype('str') + ":00", format='%Y-%m-%d %H:%M')
    summary = summary.set_index(pd.DatetimeIndex(summary['Datetime']))
    summary = summary.drop(['Date', 'Hour', 'Datetime'], axis=1)
    
    bus_detail_df = pd.read_csv(output_dir / "bus_detail.csv")
    bus_detail_df['LMP'] = bus_detail_df['LMP'].astype('float64')
    bus_detail_df['LMP DA'] = bus_detail_df['LMP DA'].astype('float64')
    bus_detail_df['Datetime'] = pd.to_datetime(bus_detail_df['Date'].astype('str') + " " + bus_detail_df['Hour'].astype('str') + ":" + bus_detail_df['Minute'].astype('str'), format='%Y-%m-%d %H:%M')
    bus_detail_df = bus_detail_df.set_index(pd.DatetimeIndex(bus_detail_df['Datetime']))
    bus_detail_df = bus_detail_df.drop(['Date', 'Hour', 'Minute', 'Datetime'], axis=1)

    summary = pd.merge(summary, bus_detail_df, how='outer', on=['Datetime'])

    renewables_df = pd.read_csv(output_dir / "renewables_detail.csv")
    thermal_df = pd.read_csv(output_dir / 'thermal_detail.csv')

    load_renewables = gen_name == None or gen_name in renewables_df.Generator.unique()
    load_thermal = gen_name == None or gen_name in thermal_df.Generator.unique()
    
    if load_renewables:
        renewables_df['Datetime'] = pd.to_datetime(renewables_df['Date'].astype('str') + " " + renewables_df['Hour'].astype('str') + ":" + renewables_df['Minute'].astype('str'), format='%Y-%m-%d %H:%M')
        renewables_df = renewables_df.set_index(pd.DatetimeIndex(renewables_df['Datetime']))
        renewables_df = renewables_df.drop(['Date', 'Hour', 'Minute', 'Datetime'], axis=1)
        gen_df = renewables_df

    if load_thermal:
        thermal_df['Datetime'] = pd.to_datetime(thermal_df['Date'].astype('str') + " " + thermal_df['Hour'].astype('str') + ":" + thermal_df['Minute'].astype('str'), format='%Y-%m-%d %H:%M')
        thermal_df = thermal_df.set_index(pd.DatetimeIndex(thermal_df['Datetime']))
        thermal_df = thermal_df.drop(['Date', 'Hour', 'Minute', 'Datetime'], axis=1)
        gen_df = thermal_df

    if load_thermal and load_renewables:
        gen_df = pd.merge(renewables_df, thermal_df, how='outer', on=['Datetime', 'Generator'])

    return summary, gen_df


def read_rts_gmlc_wind_inputs(source_dir, gen_name=None):
    """
    Read capacity factors for day ahead and real time wind forecasts. 
    The forecasts are provided as 12 periods per hour, or 288 per day, and are resampled to hourly data.
    The time index is shifted by one day relative to the other time series data, so np.roll is required at the end.

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
        rt_wind = rt_wind.mean(1)
        rt_wind = np.roll(rt_wind, 1)
        da_wind = np.roll(wind_da_df[k].values, 1)

        wind_pmax = gen_df[gen_df['GEN UID'] == k]['PMax MW'].values[0]
        wind_df[k+"-RTCF"] = rt_wind / wind_pmax
        wind_df[k+"-DACF"] = da_wind / wind_pmax
    return wind_df


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

    bus_names = pd.read_csv(source_dir / "bus.csv")
    bus_dict = {k: v for k, v in zip(bus_names['Bus ID'].values, bus_names['Bus Name'].values)}
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
    tracker_df = pd.read_csv(output_dir / "tracker_detail.csv")
    tracker_df['Datetime'] = pd.to_datetime(tracker_df['Date'].astype('str') + " " + tracker_df['Hour'].astype('str') + ":00", format='%Y-%m-%d %H:%M')
    tracker_df = tracker_df.set_index(pd.DatetimeIndex(tracker_df['Datetime']))
    tracker_df = tracker_df.drop(['Date', 'Hour', 'Datetime'], axis=1)

    tracker_model_df = pd.read_csv(output_dir / "tracking_model_detail.csv")
    tracker_model_df['Datetime'] = pd.to_datetime(tracker_model_df['Date'].astype('str') + " " + tracker_model_df['Hour'].astype('str') + ":00", format='%Y-%m-%d %H:%M')
    tracker_model_df = tracker_model_df.set_index(pd.DatetimeIndex(tracker_model_df['Datetime']))
    gen_name = tracker_model_df['Generator'].values[0]
    tracker_model_df = tracker_model_df.drop(['Date', 'Hour', 'Datetime', 'Generator'], axis=1)

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
