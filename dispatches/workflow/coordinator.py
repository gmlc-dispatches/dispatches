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

from types import ModuleType

from idaes.apps.grid_integration import DoubleLoopCoordinator as _DLC
from idaes.apps.grid_integration.utils import convert_marginal_costs_to_actual_costs


class PrescientPluginModule(ModuleType):
    def __init__(self, get_configuration, register_plugins):
        self.get_configuration = get_configuration
        self.register_plugins = register_plugins


class DoubleLoopCoordinator(_DLC):

    def register_plugins(self, context, options, plugin_config):
        super().register_plugins(context, options, plugin_config)
        
        context.register_after_get_initial_actuals_model_for_sced_callback(
            self.update_static_params
        )
        context.register_after_get_initial_actuals_model_for_simulation_actuals_callback(
            self.update_static_params
        )
        context.register_after_get_initial_forecast_model_for_ruc_callback(
            self.update_static_params
        )

    @property
    def prescient_plugin_module(self):
        return PrescientPluginModule(self.get_configuration, self.register_plugins)

    def _update_static_params(self, gen_dict):
        
        is_thermal = (
            self.bidder.bidding_model_object.model_data.generator_type == "thermal"
        )
        is_renewable = (
            self.bidder.bidding_model_object.model_data.generator_type == "renewable"
        )
        for param, value in self.bidder.bidding_model_object.model_data:
            if param == "gen_name" or value is None:
                continue
            elif (
                param in gen_dict
                and isinstance(gen_dict[param], dict)
                and gen_dict[param]["data_type"] == "time_series"
            ):
                # don't touch time varying things;
                # presumably they be updated later
                continue
            elif param == "p_cost":
                if is_thermal:
                    curve_value = convert_marginal_costs_to_actual_costs(value)
                    gen_dict[param] = {
                        "data_type": "cost_curve",
                        "cost_curve_type": "piecewise",
                        "values": curve_value,
                    }
                elif is_renewable:
                    gen_dict[param] = value
                else:
                    raise NotImplementedError(
                        "generator_type must be either 'thermal' or 'renewable'"
                    )

            else:
                gen_dict[param] = value

    def update_static_params(self, options, instance):

        gen_name = self.bidder.bidding_model_object.model_data.gen_name
        gen_dict = instance.data["elements"]["generator"][gen_name]
        self._update_static_params(gen_dict)

    def pass_static_params_to_DA(self, *args, **kwargs):
        pass

    def pass_static_params_to_RT(self, *args, **kwargs):
        pass
