
import copy
import multiprocessing
import os
import pathlib
from types import ModuleType

import pandas as pd
from pyomo.common.config import ConfigDict
from prescient.simulator import Prescient


class ParameterSweepInstance:

    def __init__(self, update_function, index):
        self.update_function = update_function
        self.index = index 

    def _register_prescient_plugins(self, context, options, plugin_config):

        self.plugin_config = plugin_config

        context.register_before_ruc_solve_callback(self._modify_DA)
        context.register_before_operations_solve_callback(self._modify_RT)

    def _get_configuration(self, key):
        return ConfigDict()

    def _modify_DA(self, prescient_options, simulator, deterministic_ruc_instance, ruc_date, ruc_hour): 
        self.update_function(deterministic_ruc_instance, self.index)

    def _modify_RT(self, prescient_options, simulator, sced_instance):
        self.update_function(sced_instance, self.index)

    def run_sweep_instance(self, prescient_options):

        module = ModuleType("ParameterSweepInstance")
        module.register_plugins = self._register_prescient_plugins
        module.get_configuration = self._get_configuration

        if "plugin" not in prescient_options:
            prescient_options["plugin"] = {}
        if "ParameterSweepInstance" in prescient_options["plugin"]:
            raise RuntimeError(f"Please specify a different name for plugin ParameterSweepInstance")
        prescient_options["plugin"]["ParameterSweepInstance"] = {"module":module}
        prescient_options["output_directory"] = prescient_options["output_directory"] + f"_index_{self.index}"

        Prescient().simulate(**prescient_options)

def parameter_sweep_runner(update_function, prescient_options, index):
    ps = ParameterSweepInstance(update_function, index)
    ps.run_sweep_instance(prescient_options)

def run_sweep(update_function, prescient_options, start, stop):
    arguments = [ (update_function, prescient_options, idx) for idx in range(start, stop) ]
    for idx in range(start, stop):
        multiprocessing.Process(target=parameter_sweep_runner, args=(update_function, prescient_options, idx)).start()

class FlattenedIndexMapper:

    def __init__(self, data):
        """
        Args:
            data (dict of lists) : the points to be sampled
        """
        self.data = copy.deepcopy(data)

        self._lengths = { k : len(v) for k,v in self.data.items() }
        self._keys = tuple(self.data.keys())
        self.number_of_points = 1
        for l in self._lengths.values():
            self.number_of_points *= l

    def get_point(self, index):
        """
        Args:
            index (int) : the index in the flattened product data
        Returns:
            value (dict of values): a sample, one from each list,
                from the input data associated with the index.
        """
        if index >= self.number_of_points:
            raise ValueError(f"Index {index} is greater than the total number of points {self.number_of_points}")
        point = {}
        for k in reversed(self._keys):
            list_index = index % self._lengths[k]
            point[k] = self.data[k][list_index]
            index = index // self._lengths[k]
        assert index == 0
        return point

    def all_points_generator(self):
        for idx in range(0,self.number_of_points):
            yield idx, self.get_point(idx)

    def __call__(self, index):
        return self.get_point(index)

    def __len__(self):
        return self.number_of_points

    def keys(self):
        yield from reversed(self._keys)


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


def summarize_results(base_directory, flattened_index_mapper, generator_name, bus_name, output_directory, other_generator_name=None, other_generator_name2=None):
    """
    Summarize Prescient runs for a single generator

    Args:
        base_directory (str) : the base directory name (without index)
        flattened_index_mapper (FlattenedIndexMapper) : The indices for the sweep
        generator_name (str) : The generator name to get the dispatch for. Looks in thermal_gens.csv and then renewable_gens.csv.
        bus_name (str) : The bus to get the LMPs for.
        output_directory (str) : The location to write the summary files to.
        other_generator_name (str) : Another generator to consolidate into the main generator
        other_generator_name2 (str) : Another generator to consolidate into the main generator

    Returns:
        None
    """

    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    param_file = os.path.join(output_directory, "sweep_parameters.csv")

    # figure out if renewable or thermal or virtual
    generator_file_names = ("thermal_detail.csv", "renewables_detail.csv", "virtual_detail.csv")
    dispatch_name_map = { "thermal_detail.csv" : "Dispatch",
                          "renewables_detail.csv" : "Output",
                          "virtual_detail.csv" : "Output",
                        }
    first_directory = base_directory+"_index_0"

    def _get_gen_df(generator_name):
        for generator_file_name in generator_file_names:
            gdf = pd.read_csv(os.path.join(first_directory, generator_file_name))["Generator"]
            if generator_name in gdf.unique():
                return generator_file_name
        else: # no break
            raise RuntimeError("Could not find output for generator "+generator_name)

    generator_file_name = _get_gen_df(generator_name)
    dispatch_name = dispatch_name_map[generator_file_name]

    if other_generator_name is not None:
        other_generator_file_name = _get_gen_df(other_generator_name)
        other_dispatch_name = dispatch_name_map[other_generator_file_name]

    if other_generator_name2 is not None:
        other_generator_file_name2 = _get_gen_df(other_generator_name2)
        other_dispatch_name2 = dispatch_name_map[other_generator_file_name2]

    with open(param_file, 'w') as csv_param_file:
        csv_param_file.write("index,"+",".join(flattened_index_mapper.keys())+"\n")

        for idx, point in flattened_index_mapper.all_points_generator():
            csv_param_file.write(f"{idx},"+",".join((f"{val}" for val in point.values()))+"\n")
            directory = base_directory+f"_index_{idx}"

            if not os.path.isfile(os.path.join(directory, "overall_simulation_output.csv")):
                raise Exception(f"For index {idx}, the simulation did not complete!")

            gdf = get_gdf(directory, generator_file_name, generator_name, dispatch_name)
            df_list = [gdf]
            RT_names = [gdf.columns[0]]
            DA_names = [gdf.columns[1]]

            if other_generator_name is not None:
                ogdf = get_gdf(directory, other_generator_file_name, other_generator_name, other_dispatch_name)
                df_list.append(ogdf)
                RT_names.append(ogdf.columns[0])
                DA_names.append(ogdf.columns[1])

            if other_generator_name2 is not None:
                ogdf2 = get_gdf(directory, other_generator_file_name2, other_generator_name2, other_dispatch_name2)
                df_list.append(ogdf2)
                RT_names.append(ogdf2.columns[0])
                DA_names.append(ogdf2.columns[1])

            bdf = prescient_output_to_df(os.path.join(directory, "bus_detail.csv"))
            bdf = bdf[bdf["Bus"] == bus_name][["Datetime","LMP","LMP DA"]]
            bdf.set_index("Datetime", inplace=True)
            df_list.append(bdf)
            RT_names.append(bdf.columns[0])
            DA_names.append(bdf.columns[1])

            odf = pd.concat(df_list, axis=1)[[*RT_names,*DA_names]]
            odf.to_csv(os.path.join(output_directory, f"sweep_results_index_{idx}.csv"))
