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

__author__ = "Radhakrishna Tumbalam Gooty"

import pandas as pd
from dispatches.util import SynHist_integration


def generate_syn_realizations(
    pkl_filename, 
    set_years, 
    n_scenarios=1, 
    n_days=365, 
    result_filename="synthetic_lmps",
    write_excel=False,
):

    syn_hist = SynHist_integration(pkl_filename)

    final_lmp_data = {}
    for s in range(1, n_scenarios+1):
        final_lmp_data[s] = {}

        # Generate synthetic history
        syn_hist_dict = syn_hist.generateSyntheticHistory("price", set_years)

        # set of clusters
        cluster_set = [i for i in syn_hist_dict["LMP"][set_years[0]].keys()]

        for y in set_years:
            # Cluster map day -> cluster
            cmap = [c for d in range(n_days) for c in cluster_set
                    if d in syn_hist_dict["cluster_map"][y][c]]
            
            # Clustered LMP signals
            LMP = syn_hist_dict["LMP"][y]
            lmp_list = []

            for d in range(n_days):
                _cluster_lmps = [v for v in LMP[cmap[d]].values()]
                lmp_list.extend(_cluster_lmps)

            final_lmp_data[s][y] = lmp_list


    if write_excel:
        filename = result_filename + ".xlsx"
        df_final_lmp_data = {}

        for s in range(1, n_scenarios+1):
            df_final_lmp_data[s] = pd.DataFrame(final_lmp_data[s])

        with pd.ExcelWriter(filename) as writer:
            for s in range(1, n_scenarios+1):
                df_final_lmp_data[s].to_excel(writer, sheet_name="Scenario" + str(s))

    if n_scenarios == 1:
        return final_lmp_data[1]
    
    else:
        return final_lmp_data
