
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

def summarize_results(base_directory, flattened_index_mapper, generator_name, bus_name, output_directory, other_generator_name=None):
    """
    Summarize Prescient runs for a single generator

    Args:
        base_directory (str) : the base directory name (without index)
        flattened_index_mapper (FlattenedIndexMapper) : The indices for the sweep
        generator_name (str) : The generator name to get the dispatch for. Looks in thermal_gens.csv and then renewable_gens.csv.
        bus_name (str) : The bus to get the LMPs for.
        output_directory (str) : The location to write the summary files to.
        other_generator_name (str) : Another generator to consolidate into the main generator

    Returns:
        None
    """

    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)

    param_file = os.path.join(output_directory, "sweep_parameters.csv")

    # figure out if renewable or thermal or virtual
    generator_file_names = ("thermal_detail.csv", "renewable_detail.csv", "virtual_detail.csv")
    first_directory = base_directory+"_index_0"

    def _get_gen_df(generator_name):
        for generator_file_name in generator_file_names:
            gdf = pd.read_csv(os.path.join(first_directory, generator_file_name))["Generator"]
            if generator_name in gdf.unique():
                return generator_file_name
        else: # no break
            raise RuntimeError("Could not find output for generator "+generator_name)

    generator_file_name = _get_gen_df(generator_name)

    if other_generator_name is not None:
        other_generator_file_name = _get_gen_df(other_generator_name)

    with open(param_file, 'w') as csv_param_file:
        csv_param_file.write("index,"+",".join(flattened_index_mapper.keys())+"\n")

        for idx, point in flattened_index_mapper.all_points_generator():
            csv_param_file.write(f"{idx},"+",".join((f"{val}" for val in point.values()))+"\n")
            directory = base_directory+f"_index_{idx}"

            gdf = prescient_output_to_df(os.path.join(directory, generator_file_name))
            gdf = gdf[gdf["Generator"] == generator_name][["Datetime","Dispatch", "Dispatch DA"]]
            gdf.set_index("Datetime", inplace=True)

            if other_generator_name is not None:
                ogdf = prescient_output_to_df(os.path.join(directory, other_generator_file_name))
                ogdf = gdf[gdf["Generator"] == other_generator_name][["Datetime","Dispatch", "Dispatch DA"]]
                ogdf.set_index("Datetime", inplace=True)

                gdf = gdf + ogdf

            bdf = prescient_output_to_df(os.path.join(directory, "bus_detail.csv"))
            bdf = bdf[bdf["Bus"] == bus_name][["Datetime","LMP","LMP DA"]]
            bdf.set_index("Datetime", inplace=True)

            odf = pd.concat([bdf,gdf], axis=1)[["Dispatch","LMP","Dispatch DA","LMP DA"]]
            odf.to_csv(os.path.join(output_directory, f"sweep_results_index_{idx}.csv"))
