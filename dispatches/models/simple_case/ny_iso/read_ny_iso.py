
#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
import numpy as np

ny_iso_df = pd.read_csv("2020_Day-Ahead_Market_Generator_LBMP.csv")
generator = "INDIAN POINT___2"
nuclear_gen_data = ny_iso_df.loc[ny_iso_df['Gen Name'] == generator]
lmp = nuclear_gen_data["DAM Gen LBMP"]
lmp_np = lmp.to_numpy()
with open('ny_iso_results.npy', 'wb') as f:
    np.save(f,lmp_np)
